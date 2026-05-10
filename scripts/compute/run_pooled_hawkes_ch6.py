"""Pooled Hawkes (no per-user multiplier) on ch.6 protocol (207d train, 52d test).

Model:
  lambda_{u,t} = c * b_t + alpha^T s_{u,t}

  c        — single global scalar
  b_t      — rolling-seasonal baseline
  s_{u,t}  — per-user Hawkes state vector (exp-decay basis over user's own
             event-feature history with half-lives 1d, 3d)
  alpha    — single global vector of weights (5 features x 2 half-lives = 10)

Total trainable parameters: 11. No per-user lambda_u.

Saves a summary.json in the format consumed by run_ladder_summary.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mpl_config = ROOT / ".mplconfig"
xdg_cache = ROOT / ".cache"
mpl_config.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.diploma_baselines.data import filter_date_range, load_daily_grid
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    build_user_states_cache,
    fit_pooled_hawkes,
)


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

# ch.6 protocol
TRAIN_START = pd.Timestamp("2025-01-15")
TRAIN_END = pd.Timestamp("2025-08-09")
TEST_START = pd.Timestamp("2025-08-10")
TEST_END = pd.Timestamp("2025-09-30")

OUTPUT_DIR = Path("diploma/reports/pooled_hawkes_ch6")


def main():
    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    train_df = filter_date_range(full_df, start_date=TRAIN_START, end_date=TRAIN_END).copy()
    test_df = filter_date_range(full_df, start_date=TEST_START, end_date=TEST_END).copy()
    print(f"  train rows: {len(train_df):,}, test rows: {len(test_df):,}")

    train_daily_mean = train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean, daily_mean_full
    )
    b_train = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    b_test = rs.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)

    print("Building Hawkes states (using only history up to train_end for test states)...")
    cache = build_user_states_cache(full_df, features=HAWKES_FEATURES, half_lives=HALF_LIVES)
    X_train = cache.gather_for(train_df)
    X_test = cache.gather_for(test_df)

    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    print("\nFitting pooled Hawkes (c, alpha)...")
    t0 = time.time()
    pooled_res = fit_pooled_hawkes(
        y=y_train, b=b_train, states=X_train.astype(float), max_iter=500,
    )
    c_fit, alpha_fit, train_loss, ok = (
        pooled_res.c, pooled_res.alpha, pooled_res.train_loss, pooled_res.converged,
    )
    print(f"  done in {time.time() - t0:.1f}s. converged={ok}, train_loss={train_loss:.2f}")
    print(f"  c = {c_fit:.6f}")
    print(f"  ||alpha|| = {np.linalg.norm(alpha_fit):.6f}")
    print(f"  alpha = {np.round(alpha_fit, 6)}")

    lam_test = np.clip(c_fit * b_test + X_test.astype(float) @ alpha_fit, 1e-8, None)
    metrics = evaluate_count_forecast(y_test, lam_test)
    print(f"\n  test mean Poisson NLL = {metrics['mean_poisson_nll']:.4f}")
    print(f"  test poisson_loglik   = {metrics['poisson_loglik']:.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "test_panel": {"rows": int(len(y_test))},
        "test_metrics_pooled_hawkes": {
            "poisson_loglik": float(metrics["poisson_loglik"]),
            "mean_poisson_nll": float(metrics["mean_poisson_nll"]),
            "mean_poisson_deviance": float(metrics["mean_poisson_deviance"]),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "aggregate_bias": float(metrics["aggregate_bias"]),
            "relative_aggregate_bias": float(metrics["relative_aggregate_bias"]),
            "mean_target": float(metrics["mean_target"]),
            "mean_prediction": float(metrics["mean_prediction"]),
        },
        "fit_params": {
            "alpha_l2": 1e-4,
            "scale_l2": 10.0,
            "c": float(c_fit),
            "alpha_norm": float(np.linalg.norm(alpha_fit)),
            "alpha": alpha_fit.tolist(),
            "feature_names": list(HAWKES_FEATURES),
            "half_lives": list(HALF_LIVES),
            "n_params": int(1 + n_alpha),
            "converged": bool(ok),
            "train_loss": float(train_loss),
        },
        "train_panel": {
            "start": str(TRAIN_START.date()),
            "end": str(TRAIN_END.date()),
            "rows": int(len(y_train)),
        },
        "test_panel_dates": {
            "start": str(TEST_START.date()),
            "end": str(TEST_END.date()),
        },
    }

    out_path = OUTPUT_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

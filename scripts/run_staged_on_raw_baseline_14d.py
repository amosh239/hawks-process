"""On each 14d block, fit staged Hawkes with RAW b_t base (no EB scaling).

Model: lambda_{u,t} = c * b_t + alpha^T s_{u,t}
where c is one global scalar, alpha is the pooled additive multi-kernel Hawkes
weights, s_{u,t} are per-user Hawkes states.

This is the "no static per-user multiplier" variant — personalization comes
entirely from per-user Hawkes states.

Compare to:
  - Personalized Gamma-Poisson (static lambda_u, no Hawkes)
  - Joint lambda_u + alpha (both static + dynamic per-user)
  - Staged with EB base (current pipeline) — degenerates to alpha=0
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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

from src.diploma_baselines.data import load_daily_grid
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    build_user_states_cache,
    fit_pooled_hawkes as _fit_pooled_hawkes,
)


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14


def fit_staged_on_raw(X, y, b_raw, alpha_l2=1e-4, scale_l2=10.0, max_iter=400):
    """Thin wrapper around the canonical `fit_pooled_hawkes` returning legacy `(c, α, ok)`."""
    res = _fit_pooled_hawkes(
        y=y, b=b_raw, states=X,
        alpha_l2=alpha_l2, scale_l2=scale_l2, max_iter=max_iter,
    )
    return res.c, res.alpha, res.converged


def main():
    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    print("\nBuilding Hawkes states from full history...")
    cache = build_user_states_cache(full_df, features=HAWKES_FEATURES, half_lives=HALF_LIVES)
    print(f"  n_alpha = {cache.n_alpha}")

    blocks = []
    cursor = CV_GLOBAL_START
    idx = 0
    while True:
        block_start = cursor
        block_end = cursor + pd.Timedelta(days=BLOCK_LEN - 1)
        if block_end > CV_GLOBAL_END:
            break
        train_end = block_start + pd.Timedelta(days=TRAIN_LEN - 1)
        blocks.append({"block_idx": idx, "block_start": block_start, "train_end": train_end,
                       "block_end": block_end})
        idx += 1
        cursor = block_end + pd.Timedelta(days=1)

    rows: list[dict] = []
    for block in blocks:
        block_start = block["block_start"]
        train_end = block["train_end"]
        block_end = block["block_end"]
        test_start = train_end + pd.Timedelta(days=1)

        block_train_df = full_df.loc[
            (full_df["event_date"] >= block_start) & (full_df["event_date"] <= train_end)
        ].copy()
        block_test_df = full_df.loc[
            (full_df["event_date"] >= test_start) & (full_df["event_date"] <= block_end)
        ].copy()

        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean, daily_mean_full
        )
        b_train = rs.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)
        b_test = rs.predict_for_dates(block_test_df["event_date"]).to_numpy(dtype=float)

        X_train = cache.gather_for(block_train_df).astype(float)
        X_test = cache.gather_for(block_test_df).astype(float)
        y_train = block_train_df[TARGET_COL].to_numpy(dtype=float)
        y_test = block_test_df[TARGET_COL].to_numpy(dtype=float)

        c_fit, alpha_fit, ok = fit_staged_on_raw(X_train, y_train, b_train)
        alpha_norm = float(np.linalg.norm(alpha_fit))

        # Test NLL
        lam_test = np.clip(c_fit * b_test + X_test @ alpha_fit, 1e-8, None)
        nll_test = float(evaluate_count_forecast(y_test, lam_test)["mean_poisson_nll"])

        # Reference: c-only on raw (no Hawkes), should match Rolling Seasonal NLL when c=1.
        from scipy.optimize import minimize_scalar
        def loss_c(c):
            lam = np.clip(float(c) * b_train, 1e-8, None)
            return float(np.sum(lam - y_train * np.log(lam)))
        res_c = minimize_scalar(loss_c, bounds=(0.001, 10.0), method="bounded")
        c_only = float(res_c.x)
        lam_test_c_only = np.clip(c_only * b_test, 1e-8, None)
        nll_c_only = float(evaluate_count_forecast(y_test, lam_test_c_only)["mean_poisson_nll"])

        rows.append(
            {
                "block_idx": block["block_idx"],
                "block_label": f"{block_start.date()}..{block_end.date()}",
                "test_rows": int(len(block_test_df)),
                "Staged-on-raw (c + alpha)": nll_test,
                "Global-c-on-raw (no Hawkes)": nll_c_only,
                "c_fit": c_fit,
                "alpha_norm": alpha_norm,
                "c_only_fit": c_only,
            }
        )
        print(
            f"  block {block['block_idx'] + 1:>2}/13 {block_start.date()}..{train_end.date()}  "
            f"c={c_fit:.3f} ||α||={alpha_norm:.4f}  "
            f"NLL: c+α={nll_test:.4f}  c-only={nll_c_only:.4f}  Δ={nll_test - nll_c_only:+.4f}"
        )

    out_path = Path("diploma/reports/blockwise_cv/staged_on_raw_14d.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved per-block CSV to {out_path}")

    print("\n=== Summary ===")
    for col in ["Staged-on-raw (c + alpha)", "Global-c-on-raw (no Hawkes)"]:
        vals = df[col].to_numpy(dtype=float)
        print(f"  {col:<32s}  mean={vals.mean():.4f}  median={np.median(vals):.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

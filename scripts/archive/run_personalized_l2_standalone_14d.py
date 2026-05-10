"""True standalone Personalized L2 fit on each 14d block.

Unlike `run_joint_fit_on_14d_block.py` (which fits λ_u jointly with α and then
predicts without α at test-time), this script fits ONLY λ_u — no α anywhere,
no Hawkes states. The L2 prior is identical (γ=1, shrinkage to 1, bounds
[0.001, 50]).

This is the "fair" Personalized L2 baseline that:
  - matches the chapter-8 ladder formulation (true standalone L2),
  - can be compared apples-to-apples with Personalized Gamma-Poisson on 14d,
  - measures the actual contribution of L2-vs-Gamma regularization on 14d
    without any influence from α-side effects in joint optimization.

Outputs a CSV with per-block test NLL, used by chapter 15 and the updated
chapter-10 strip plot.
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
from src.diploma_baselines.models import GlobalRollingSeasonalPoissonModel


TARGET_COL = "to_ord"
WINDOW_SIZE = 7

CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14
GAMMA_L2 = 1.0


def fit_personalized_l2_only(user_idx_arr, y, b, n_users, gamma):
    """Pure standalone Personalized L2 — no alpha, no Hawkes, only λ_u with L2(γ).

    Solves: argmin_{λ_u} Σ_t (λ_u(t) · b_t − y_t · log(λ_u(t) · b_t))
                        + γ · Σ_u (λ_u − 1)²
    subject to λ_u ∈ [0.001, 50].
    """
    init_lam = np.ones(n_users)
    bounds = [(0.001, 50.0)] * n_users

    def _obj(params):
        mu = np.clip(params[user_idx_arr] * b, 1e-8, None)
        nll = float(
            np.sum(mu - y * np.log(mu))
            + gamma * np.sum((params - 1.0) ** 2)
        )
        residual = 1.0 - y / mu
        grad = np.zeros(n_users)
        np.add.at(grad, user_idx_arr, residual * b)
        grad += 2.0 * gamma * (params - 1.0)
        return nll, grad

    res = minimize(_obj, init_lam, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 400})
    return res.x


from src.diploma_baselines.metrics import evaluate_count_forecast


def standard_poisson_nll(y: np.ndarray, lam: np.ndarray) -> float:
    """Full Poisson NLL via evaluate_count_forecast (includes log(y!) term)."""
    return float(evaluate_count_forecast(np.asarray(y, dtype=float), np.asarray(lam, dtype=float))["mean_poisson_nll"])


def main() -> None:
    print("Loading data...")
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=[TARGET_COL],
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    blocks = []
    cursor = CV_GLOBAL_START
    idx = 0
    while True:
        block_start = cursor
        block_end = cursor + pd.Timedelta(days=BLOCK_LEN - 1)
        if block_end > CV_GLOBAL_END:
            break
        train_end = block_start + pd.Timedelta(days=TRAIN_LEN - 1)
        blocks.append({
            "block_idx": idx,
            "block_start": block_start,
            "train_end": train_end,
            "test_start": train_end + pd.Timedelta(days=1),
            "test_end": block_end,
        })
        idx += 1
        cursor = block_end + pd.Timedelta(days=1)

    print(f"\n=== True standalone Personalized L2 fit on each of {len(blocks)} blocks (γ={GAMMA_L2}) ===\n")

    rows = []
    for block in blocks:
        block_start = block["block_start"]
        train_end = block["train_end"]
        test_start = block["test_start"]
        test_end = block["test_end"]

        block_train_df = full_df.loc[
            (full_df["event_date"] >= block_start) & (full_df["event_date"] <= train_end)
        ].copy()
        block_test_df = full_df.loc[
            (full_df["event_date"] >= test_start) & (full_df["event_date"] <= test_end)
        ].copy()

        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs_block = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean, daily_mean_full
        )
        base_train = rs_block.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)
        base_test = rs_block.predict_for_dates(block_test_df["event_date"]).to_numpy(dtype=float)

        train_uids = block_train_df["user_id"].to_numpy()
        test_uids = block_test_df["user_id"].to_numpy()
        unique_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
        n_users = int(len(unique_uids))
        y_train = block_train_df[TARGET_COL].to_numpy(dtype=float)
        y_test = block_test_df[TARGET_COL].to_numpy(dtype=float)

        # === True standalone L2 fit ===
        lam_u_fit = fit_personalized_l2_only(train_user_idx, y_train, base_train, n_users, GAMMA_L2)

        # Predict on test
        uid_to_idx = {int(u): i for i, u in enumerate(unique_uids)}
        test_user_idx = np.array([uid_to_idx.get(int(u), -1) for u in test_uids], dtype=np.int64)
        lam_u_for_test = np.where(test_user_idx >= 0, lam_u_fit[np.maximum(test_user_idx, 0)], 1.0)
        lam_test = np.clip(lam_u_for_test * base_test, 1e-8, None)
        nll = standard_poisson_nll(y_test, lam_test)

        rows.append({
            "block_idx": block["block_idx"],
            "block_label": f"{block_start.date()}..{block['block_start'] + pd.Timedelta(days=BLOCK_LEN - 1):%Y-%m-%d}",
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "test_rows": int(len(block_test_df)),
            "Personalized L2 standalone": nll,
            "lambda_mean": float(lam_u_fit.mean()),
            "lambda_median": float(np.median(lam_u_fit)),
        })

        print(
            f"  block {block['block_idx'] + 1:>2}/13 {block_start.date()}..{train_end.date()}  "
            f"NLL={nll:.4f}  λ_u: mean={lam_u_fit.mean():.3f}  median={np.median(lam_u_fit):.3f}"
        )

    out_path = Path("diploma/reports/joint_lambda_alpha/personalized_l2_standalone_14d.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nMean NLL across 13 blocks = {df['Personalized L2 standalone'].mean():.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

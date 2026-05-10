"""Quick check: does joint lambda+alpha MLE also degenerate on 14d blocks
like staged EB+Hawkes does?

Also saves per-block test NLL CSV used by the chapter 10 strip plot
re-generation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd  # for type hints, real import below

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

from src.diploma_baselines.data import load_daily_grid
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    build_user_states_cache,
)

from scripts.run_joint_lambda_alpha_fit import fit_joint  # type: ignore
from src.diploma_baselines.metrics import evaluate_count_forecast


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14


def main() -> None:
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
    n_alpha = cache.n_alpha

    blocks = []
    cursor = CV_GLOBAL_START
    idx = 0
    while True:
        block_start = cursor
        block_end = cursor + pd.Timedelta(days=BLOCK_LEN - 1)
        if block_end > CV_GLOBAL_END:
            break
        train_end = block_start + pd.Timedelta(days=TRAIN_LEN - 1)
        blocks.append({"block_idx": idx, "block_start": block_start, "train_end": train_end})
        idx += 1
        cursor = block_end + pd.Timedelta(days=1)

    print(f"\n=== Joint lambda+alpha MLE on 14d train of each block ===")

    rows: list[dict] = []
    for block in blocks:
        block_start = block["block_start"]
        train_end = block["train_end"]

        block_train_df = full_df.loc[
            (full_df["event_date"] >= block_start) & (full_df["event_date"] <= train_end)
        ].copy()

        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs_block = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean, daily_mean_full
        )
        base_train = rs_block.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)

        states_train = cache.gather_for(block_train_df)

        train_uids = block_train_df["user_id"].to_numpy()
        unique_train_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
        n_users = int(len(unique_train_uids))
        y_train = block_train_df[TARGET_COL].to_numpy(dtype=float)

        # Joint fit with mild shrinkage (lambda_l2 = 1 — best on ch.6)
        lam_u_fit, alpha_fit, info_fit = fit_joint(
            user_idx=train_user_idx,
            y=y_train,
            b=base_train,
            states=states_train.astype(float),
            n_users=n_users,
            n_alpha=n_alpha,
            lambda_l2=1.0,
            alpha_l2=1e-4,
            max_iter=400,
            verbose=False,
        )
        alpha_norm = float(np.linalg.norm(alpha_fit))

        # Evaluate on the 7d test of the same block
        test_start = train_end + pd.Timedelta(days=1)
        test_end = block_start + pd.Timedelta(days=BLOCK_LEN - 1)
        block_test_df = full_df.loc[
            (full_df["event_date"] >= test_start) & (full_df["event_date"] <= test_end)
        ].copy()

        base_test = rs_block.predict_for_dates(block_test_df["event_date"]).to_numpy(dtype=float)
        states_test = cache.gather_for(block_test_df)

        test_uids = block_test_df["user_id"].to_numpy()
        uid_to_idx = {int(u): i for i, u in enumerate(unique_train_uids)}
        test_user_idx = np.array([uid_to_idx.get(int(u), -1) for u in test_uids], dtype=np.int64)
        lam_u_for_test = np.where(test_user_idx >= 0, lam_u_fit[np.maximum(test_user_idx, 0)], 1.0)
        lam_test = lam_u_for_test * base_test + states_test.astype(float) @ alpha_fit
        y_test = block_test_df[TARGET_COL].to_numpy(dtype=float)
        nll_test = float(evaluate_count_forecast(y_test, lam_test)["mean_poisson_nll"])

        # Reference: lambda only (alpha=0)
        lam_test_lambda_only = lam_u_for_test * base_test
        nll_test_lambda_only = float(evaluate_count_forecast(y_test, lam_test_lambda_only)["mean_poisson_nll"])

        rows.append(
            {
                "block_idx": block["block_idx"],
                "block_label": f"{block_start.date()}..{block_start + pd.Timedelta(days=BLOCK_LEN - 1):%Y-%m-%d}",
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "test_rows": int(len(block_test_df)),
                "Joint Hawkes (λ_u + α)": nll_test,
                "Joint baseline (λ_u only)": nll_test_lambda_only,
                "alpha_norm": alpha_norm,
                "lambda_mean": float(lam_u_fit.mean()),
            }
        )

        print(
            f"  block {block['block_idx'] + 1:>2}/13 {block_start.date()}..{train_end.date()}  "
            f"||α||={alpha_norm:.4f}  λ_u mean={lam_u_fit.mean():.3f}  "
            f"test NLL: λ+α={nll_test:.4f}  λ-only={nll_test_lambda_only:.4f}  Δ={nll_test - nll_test_lambda_only:+.4f}"
        )

    out_path = Path("diploma/reports/joint_lambda_alpha/joint_14d_per_block.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved per-block CSV to {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

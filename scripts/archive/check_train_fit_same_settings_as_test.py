"""Sanity check: run Hawkes fit on TRAIN with the EXACT settings used for test fit
in chapter 13 (alpha_l2=0, scale_l2=0, max_iter=500, both default and warm-start
inits). Make sure the train-fit really does go to (c=1, alpha=0) under these
identical settings.
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

from src.diploma_baselines.data import load_daily_grid
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_basis_states,
    fit_pooled_additive_multi_kernel_hawkes,
)


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14

LONG_TRAIN_ALPHA = np.array(
    [
        [0.003, 0.000],
        [0.001, 0.000],
        [0.000, 0.001],
        [0.004, 0.000],
        [0.000, 0.015],
    ],
    dtype=float,
).reshape(-1)
LONG_TRAIN_SCALE = 0.8262


def main() -> None:
    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full_target = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    beta = np.log(2.0) / np.asarray(HALF_LIVES, dtype=float)
    print("\nBuilding Hawkes states from full history...")
    user_states: dict[int, dict] = {}
    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        user_states[int(user_id)] = {
            "states_full": states_full,
            "dates": full_user["event_date"].to_numpy(dtype="datetime64[ns]"),
        }
    print(f"  states for {len(user_states):,} users")

    blocks = []
    cursor = CV_GLOBAL_START
    idx = 0
    while True:
        block_start = cursor
        block_end = cursor + pd.Timedelta(days=BLOCK_LEN - 1)
        if block_end > CV_GLOBAL_END:
            break
        train_end = block_start + pd.Timedelta(days=TRAIN_LEN - 1)
        blocks.append({"block_idx": idx, "block_start": block_start, "train_end": train_end, "block_end": block_end})
        idx += 1
        cursor = block_end + pd.Timedelta(days=1)

    print("\n=== Fit Hawkes on TRAIN with EXACT same settings as test fit (alpha_l2=0, scale_l2=0, max_iter=500) ===\n")
    for block in blocks:
        block_start = block["block_start"]
        train_end = block["train_end"]

        block_train_df = full_df.loc[
            (full_df["event_date"] >= block_start) & (full_df["event_date"] <= train_end)
        ].copy()

        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs_block = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean, daily_mean_full_target
        )
        base_train = rs_block.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)
        scaler_block = PersonalizedGammaPoissonScaler().fit(
            block_train_df["user_id"].to_numpy(),
            block_train_df[TARGET_COL].to_numpy(),
            base_train,
        )
        pers_train = scaler_block.predict(
            block_train_df["user_id"].to_numpy(), base_train, method="posterior_mean"
        )

        train_dates = block_train_df["event_date"].to_numpy(dtype="datetime64[ns]")
        state_blocks = []
        y_blocks = []
        base_blocks = []
        for uid, idx_in_block in block_train_df.groupby("user_id", sort=False).indices.items():
            info = user_states[int(uid)]
            full_dates = info["dates"]
            wanted_dates = train_dates[idx_in_block]
            full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}
            rows_in_full = np.array(
                [full_to_idx[pd.Timestamp(d).normalize()] for d in wanted_dates], dtype=int
            )
            user_states_train = info["states_full"][rows_in_full]
            state_blocks.append(user_states_train)
            y_blocks.append(block_train_df[TARGET_COL].to_numpy(dtype=float)[idx_in_block])
            base_blocks.append(pers_train[idx_in_block])

        # Default init
        fit_default = fit_pooled_additive_multi_kernel_hawkes(
            state_blocks=state_blocks,
            y_blocks=y_blocks,
            base_blocks=base_blocks,
            half_lives=HALF_LIVES,
            feature_names=HAWKES_FEATURES,
            alpha_l2=0.0,
            learn_base_scale=True,
            scale_l2=0.0,
            scale_init=1.0,
            max_iter=500,
        )

        # Warm-start ch.6
        fit_warm = fit_pooled_additive_multi_kernel_hawkes(
            state_blocks=state_blocks,
            y_blocks=y_blocks,
            base_blocks=base_blocks,
            half_lives=HALF_LIVES,
            feature_names=HAWKES_FEATURES,
            alpha_l2=0.0,
            learn_base_scale=True,
            scale_l2=0.0,
            scale_init=LONG_TRAIN_SCALE,
            alpha_init=LONG_TRAIN_ALPHA,
            max_iter=500,
        )

        norm_def = float(np.linalg.norm(fit_default.alpha))
        norm_warm = float(np.linalg.norm(fit_warm.alpha))
        print(
            f"  block {block['block_idx'] + 1:>2}/13 {block_start.date()}..{train_end.date()}  "
            f"default-init c={fit_default.base_scale:.4f} ||a||={norm_def:.4f}  |  "
            f"warm-start  c={fit_warm.base_scale:.4f} ||a||={norm_warm:.4f}"
        )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

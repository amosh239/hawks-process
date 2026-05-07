"""Quick sanity check: does frozen ch.6 (c, alpha) really have higher train NLL
than (c=1, alpha=0) on each 14d block?

Chapter 11 *claims* (c=1, alpha=0) is the train-loss optimum on 14d. But that
claim is based on optimizer behaviour. Here we check the train NLL at the two
points DIRECTLY without involving the optimizer at all, on every 14d block.
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

from src.diploma_baselines.data import filter_date_range, load_daily_grid
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_basis_states,
)


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14

# Frozen ch.6 Hawkes params
LONG_TRAIN_ALPHA = np.array(
    [
        [0.003, 0.000],   # searches
        [0.001, 0.000],   # cat_to_cart
        [0.000, 0.001],   # cat_to_ord
        [0.004, 0.000],   # to_cart
        [0.000, 0.015],   # to_ord
    ],
    dtype=float,
).reshape(-1)
LONG_TRAIN_SCALE = 0.8262


def poisson_nll_per_obs(y: np.ndarray, lam: np.ndarray) -> float:
    lam = np.clip(lam, 1e-8, None)
    return float(np.mean(lam - y * np.log(np.maximum(y, 1e-300)) + y * np.log(y / lam) - y + lam - lam))


def standard_poisson_nll(y: np.ndarray, lam: np.ndarray) -> float:
    """Standard mean Poisson NLL: mean over rows of (lam - y*log(lam))."""
    lam = np.clip(lam, 1e-8, None)
    return float(np.mean(lam - y * np.log(lam)))


def main() -> None:
    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full_target = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    # Build full-history Hawkes states for every user once (warmup is implicit)
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

    # Build all 13 blocks
    blocks: list[dict] = []
    cursor = CV_GLOBAL_START
    idx = 0
    while True:
        block_start = cursor
        block_end = cursor + pd.Timedelta(days=BLOCK_LEN - 1)
        if block_end > CV_GLOBAL_END:
            break
        train_end = block_start + pd.Timedelta(days=TRAIN_LEN - 1)
        blocks.append(
            {
                "block_idx": idx,
                "block_start": block_start,
                "block_end": block_end,
                "train_end": train_end,
            }
        )
        idx += 1
        cursor = block_end + pd.Timedelta(days=1)
    print(f"  built {len(blocks)} blocks")

    print("\n=== Train NLL at (c=1, alpha=0) vs (c=0.8262, alpha_ch6) on each 14d train block ===\n")
    print(f"  alpha_ch6 = {LONG_TRAIN_ALPHA}")
    print(f"  ||alpha_ch6||_2 = {np.linalg.norm(LONG_TRAIN_ALPHA):.4f}\n")

    results = []
    for block in blocks:
        block_start = block["block_start"]
        train_end = block["train_end"]

        block_train_df = full_df.loc[
            (full_df["event_date"] >= block_start) & (full_df["event_date"] <= train_end)
        ].copy()

        # Block-fit personalized baseline (same as setting A / E in 11c)
        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs_block = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean,
            daily_mean_full_target,
        )
        base_train = rs_block.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)
        scaler_block = PersonalizedGammaPoissonScaler().fit(
            block_train_df["user_id"].to_numpy(),
            block_train_df[TARGET_COL].to_numpy(),
            base_train,
        )
        pers_train = scaler_block.predict(
            block_train_df["user_id"].to_numpy(),
            base_train,
            method="posterior_mean",
        )

        # For Hawkes, gather state rows from full-history states (warmed up)
        train_dates = block_train_df["event_date"].to_numpy(dtype="datetime64[ns]")
        all_states = np.zeros((len(block_train_df), LONG_TRAIN_ALPHA.shape[0]), dtype=float)
        for uid, idx_in_block in block_train_df.groupby("user_id", sort=False).indices.items():
            info = user_states[int(uid)]
            full_dates = info["dates"]
            wanted_dates = train_dates[idx_in_block]
            full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}
            rows_in_full = np.array(
                [full_to_idx[pd.Timestamp(d).normalize()] for d in wanted_dates],
                dtype=int,
            )
            all_states[idx_in_block] = info["states_full"][rows_in_full]

        y_train = block_train_df[TARGET_COL].to_numpy(dtype=float)

        # Lambda at the two parameter points
        lam_degenerate = 1.0 * pers_train  # c=1, alpha=0 => lam = pers
        lam_frozen = LONG_TRAIN_SCALE * pers_train + all_states @ LONG_TRAIN_ALPHA

        nll_degenerate = standard_poisson_nll(y_train, lam_degenerate)
        nll_frozen = standard_poisson_nll(y_train, lam_frozen)
        delta = nll_frozen - nll_degenerate

        results.append(
            {
                "block_idx": block["block_idx"],
                "label": f"{block_start.date()}..{train_end.date()}",
                "n_train": int(len(block_train_df)),
                "nll_degenerate": nll_degenerate,
                "nll_frozen": nll_frozen,
                "delta": delta,
            }
        )

        marker = "[BAD]" if delta < -1e-6 else "[OK ]"  # frozen lower than degenerate => 11 wrong
        print(
            f"  {marker} block {block['block_idx'] + 1:>2}/13 {block_start.date()}..{train_end.date()}  "
            f"nll(degen)={nll_degenerate:.6f}  nll(frozen ch.6)={nll_frozen:.6f}  "
            f"delta={delta:+.6f}"
        )

    deltas = np.array([r["delta"] for r in results])
    print(
        f"\nSummary: delta = nll(frozen) - nll(degenerate) over 13 blocks:\n"
        f"  mean   = {deltas.mean():+.6f}\n"
        f"  median = {np.median(deltas):+.6f}\n"
        f"  min    = {deltas.min():+.6f}\n"
        f"  max    = {deltas.max():+.6f}\n"
        f"  blocks where frozen has LOWER train NLL: {int((deltas < -1e-6).sum())} / 13"
    )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

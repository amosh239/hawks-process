"""Fit Hawkes directly on the 7d TEST data of each CV block.

For each of the 13 three-week blocks (chapter 10), fit Hawkes on the 7d test
window (with block-fit personalized as base, warmed-up states from full history).
Try two initializations:

  init_default:   alpha=0.01, scale=1
  init_ch6:       alpha=alpha_ch6, scale=0.8262 (warm-start from chapter 6)

Record the converged (c, alpha) and test NLL. Compare with degenerate point
(c=1, alpha=0) and frozen ch.6 point.
"""

from __future__ import annotations

import json
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diploma_baselines.data import load_daily_grid
from src.diploma_baselines.metrics import evaluate_count_forecast
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

OUTPUT_DIR = Path("diploma/reports/hawkes_fit_on_test")


def standard_poisson_nll(y: np.ndarray, lam: np.ndarray) -> float:
    lam = np.clip(lam, 1e-8, None)
    return float(np.mean(lam - y * np.log(lam)))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
                "test_start": train_end + pd.Timedelta(days=1),
                "test_end": block_end,
            }
        )
        idx += 1
        cursor = block_end + pd.Timedelta(days=1)

    print(f"\n=== Fitting Hawkes on 7d TEST of each block (2 inits per block) ===\n")

    results = []
    for block in blocks:
        block_start = block["block_start"]
        train_end = block["train_end"]
        test_start = block["test_start"]
        test_end = block["test_end"]

        # Block-fit personalized (same protocol as setting A/E in chapter 11c)
        block_train_df = full_df.loc[
            (full_df["event_date"] >= block_start) & (full_df["event_date"] <= train_end)
        ].copy()
        block_test_df = full_df.loc[
            (full_df["event_date"] >= test_start) & (full_df["event_date"] <= test_end)
        ].copy()

        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs_block = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean,
            daily_mean_full_target,
        )
        base_train = rs_block.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)
        base_test = rs_block.predict_for_dates(block_test_df["event_date"]).to_numpy(dtype=float)
        scaler_block = PersonalizedGammaPoissonScaler().fit(
            block_train_df["user_id"].to_numpy(),
            block_train_df[TARGET_COL].to_numpy(),
            base_train,
        )
        pers_test = scaler_block.predict(
            block_test_df["user_id"].to_numpy(), base_test, method="posterior_mean"
        )

        # Test states (warmed up from full history)
        test_dates = block_test_df["event_date"].to_numpy(dtype="datetime64[ns]")
        all_states = np.zeros((len(block_test_df), LONG_TRAIN_ALPHA.shape[0]), dtype=np.float32)
        state_blocks = []
        y_blocks = []
        base_blocks = []
        for uid, idx_in_block in block_test_df.groupby("user_id", sort=False).indices.items():
            info = user_states[int(uid)]
            full_dates = info["dates"]
            wanted_dates = test_dates[idx_in_block]
            full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}
            rows_in_full = np.array(
                [full_to_idx[pd.Timestamp(d).normalize()] for d in wanted_dates],
                dtype=int,
            )
            user_states_test = info["states_full"][rows_in_full]
            all_states[idx_in_block] = user_states_test
            state_blocks.append(user_states_test)
            y_blocks.append(block_test_df[TARGET_COL].to_numpy(dtype=float)[idx_in_block])
            base_blocks.append(pers_test[idx_in_block])

        y_test = block_test_df[TARGET_COL].to_numpy(dtype=float)

        # Reference points
        lam_degen = pers_test
        lam_frozen = LONG_TRAIN_SCALE * pers_test + all_states.astype(float) @ LONG_TRAIN_ALPHA
        nll_degen = standard_poisson_nll(y_test, lam_degen)
        nll_frozen = standard_poisson_nll(y_test, lam_frozen)

        # Fit on test, default init (alpha=0.01, scale=1)
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
        lam_default = fit_default.base_scale * pers_test + all_states.astype(float) @ fit_default.alpha
        nll_default = standard_poisson_nll(y_test, lam_default)

        # Fit on test, warm-start from ch.6
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
        lam_warm = fit_warm.base_scale * pers_test + all_states.astype(float) @ fit_warm.alpha
        nll_warm = standard_poisson_nll(y_test, lam_warm)

        row = {
            "block_idx": block["block_idx"],
            "label": f"{block_start.date()}..{block_end.date()}",
            "test_label": f"{test_start.date()}..{test_end.date()}",
            "n_test": int(len(block_test_df)),
            "nll_degenerate": nll_degen,
            "nll_frozen_ch6": nll_frozen,
            "nll_test_fit_default": nll_default,
            "nll_test_fit_warm": nll_warm,
            "test_fit_default_c": float(fit_default.base_scale),
            "test_fit_default_alpha": fit_default.alpha.tolist(),
            "test_fit_default_alpha_norm": float(np.linalg.norm(fit_default.alpha)),
            "test_fit_warm_c": float(fit_warm.base_scale),
            "test_fit_warm_alpha": fit_warm.alpha.tolist(),
            "test_fit_warm_alpha_norm": float(np.linalg.norm(fit_warm.alpha)),
        }
        results.append(row)

        print(
            f"  block {block['block_idx'] + 1:>2}/13 "
            f"deg={nll_degen:.4f}  frozen={nll_frozen:.4f}  "
            f"fit-def={nll_default:.4f} (c={fit_default.base_scale:.3f}, ||a||={row['test_fit_default_alpha_norm']:.4f})  "
            f"fit-warm={nll_warm:.4f} (c={fit_warm.base_scale:.3f}, ||a||={row['test_fit_warm_alpha_norm']:.4f})"
        )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "hawkes_fit_on_test_per_block.csv", index=False)

    # Aggregate
    summary = {
        "n_blocks": int(len(df)),
        "mean_nll_degenerate": float(df["nll_degenerate"].mean()),
        "mean_nll_frozen_ch6": float(df["nll_frozen_ch6"].mean()),
        "mean_nll_test_fit_default": float(df["nll_test_fit_default"].mean()),
        "mean_nll_test_fit_warm": float(df["nll_test_fit_warm"].mean()),
        "mean_alpha_norm_test_fit_default": float(df["test_fit_default_alpha_norm"].mean()),
        "mean_alpha_norm_test_fit_warm": float(df["test_fit_warm_alpha_norm"].mean()),
        "mean_c_test_fit_default": float(df["test_fit_default_c"].mean()),
        "mean_c_test_fit_warm": float(df["test_fit_warm_c"].mean()),
        "ch6_alpha_norm": float(np.linalg.norm(LONG_TRAIN_ALPHA)),
        "ch6_scale": LONG_TRAIN_SCALE,
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nMean test NLL across 13 blocks:")
    print(f"  degenerate (c=1, alpha=0):   {summary['mean_nll_degenerate']:.4f}")
    print(f"  frozen ch.6:                 {summary['mean_nll_frozen_ch6']:.4f}")
    print(f"  test-fit, default init:      {summary['mean_nll_test_fit_default']:.4f}")
    print(f"  test-fit, warm-start ch.6:   {summary['mean_nll_test_fit_warm']:.4f}")
    print(
        f"\nMean ||alpha||_2: ch.6={summary['ch6_alpha_norm']:.4f}  "
        f"test-fit-default={summary['mean_alpha_norm_test_fit_default']:.4f}  "
        f"test-fit-warm={summary['mean_alpha_norm_test_fit_warm']:.4f}"
    )

    # === Plot 1: per-block test NLL at 4 points ===
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    settings = [
        ("c=1, α=0", "nll_degenerate", "#888888"),
        ("frozen ch.6", "nll_frozen_ch6", "#D2691E"),
        ("test-fit (default init)", "nll_test_fit_default", "#2E5EAA"),
        ("test-fit (warm ch.6 init)", "nll_test_fit_warm", "#2E8B57"),
    ]
    x_idx = np.arange(len(df))
    width = 0.20
    for i, (label, col, color) in enumerate(settings):
        ax.bar(x_idx + (i - 1.5) * width, df[col].to_numpy(dtype=float), width, label=label, color=color, edgecolor="white")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"B{i + 1}" for i in range(len(df))])
    ax.set_ylabel("Test NLL per user-day")
    ax.set_title("Per-block test NLL: degenerate point vs frozen ch.6 vs Hawkes fit on test data")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_nll_per_block.png", dpi=150)
    plt.close(fig)

    # === Plot 2: ||alpha||_2 by init per block ===
    fig, ax = plt.subplots(figsize=(11.6, 4.4))
    ax.axhline(np.linalg.norm(LONG_TRAIN_ALPHA), color="#D2691E", linewidth=1.5, linestyle="--", label="||α||_2 ch.6 = 0.0159")
    ax.bar(x_idx - 0.20, df["test_fit_default_alpha_norm"].to_numpy(dtype=float), 0.40, label="default init", color="#2E5EAA", edgecolor="white")
    ax.bar(x_idx + 0.20, df["test_fit_warm_alpha_norm"].to_numpy(dtype=float), 0.40, label="warm-start ch.6", color="#2E8B57", edgecolor="white")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"B{i + 1}" for i in range(len(df))])
    ax.set_ylabel("Fitted ||α||_2")
    ax.set_title("Fitted ||α||_2 from Hawkes optimization on 7d TEST data, two inits")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "alpha_norm_per_block.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

"""Plot per-block multiplier on season_poisson for staged vs joint Hawkes.

For each of 13 CV blocks (chapter 10), shows:
  - staged Hawkes: c = 1.000 on every block (degenerate)
  - joint Hawkes: per-user lambda_u distribution (mean, median, p5..p95 band)

Reads:
  diploma/reports/joint_lambda_alpha/joint_14d_per_block.csv  -- per-block lambda_mean
  Per-block lambda_u distributions are computed by re-running the joint fits
  using the same protocol as run_joint_fit_on_14d_block.py, but storing the
  full per-user lambda arrays.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diploma_baselines.data import load_daily_grid
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    build_basis_states,
)

from scripts.run_joint_lambda_alpha_fit import fit_joint  # type: ignore


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14

OUTPUT_DIR = Path("diploma/reports/blockwise_cv")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    beta = np.log(2.0) / np.asarray(HALF_LIVES, dtype=float)

    print("\nBuilding Hawkes states from full history...")
    user_states_per_id: dict[int, dict] = {}
    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        user_states_per_id[int(user_id)] = {
            "states_full": states_full,
            "dates": full_user["event_date"].to_numpy(dtype="datetime64[ns]"),
        }
    n_alpha = next(iter(user_states_per_id.values()))["states_full"].shape[1]

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

    print("\n=== Refitting joint MLE on each 14d block (γ=1) to recover per-user λ_u ===")
    per_block_lambdas: list[np.ndarray] = []
    block_means: list[float] = []
    block_medians: list[float] = []
    block_p5: list[float] = []
    block_p95: list[float] = []
    block_p25: list[float] = []
    block_p75: list[float] = []
    block_alpha_norms: list[float] = []
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

        train_dates = block_train_df["event_date"].to_numpy(dtype="datetime64[ns]")
        states_train = np.zeros((len(block_train_df), n_alpha), dtype=np.float32)
        for uid, idx_in_block in block_train_df.groupby("user_id", sort=False).indices.items():
            info = user_states_per_id[int(uid)]
            full_dates = info["dates"]
            wanted_dates = train_dates[idx_in_block]
            full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}
            rows_in_full = np.array(
                [full_to_idx[pd.Timestamp(d).normalize()] for d in wanted_dates], dtype=int
            )
            states_train[idx_in_block] = info["states_full"][rows_in_full]

        train_uids = block_train_df["user_id"].to_numpy()
        unique_train_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
        n_users = int(len(unique_train_uids))
        y_train = block_train_df[TARGET_COL].to_numpy(dtype=float)

        lam_u_fit, alpha_fit, _ = fit_joint(
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
        per_block_lambdas.append(lam_u_fit)
        block_means.append(float(lam_u_fit.mean()))
        block_medians.append(float(np.median(lam_u_fit)))
        block_p5.append(float(np.percentile(lam_u_fit, 5)))
        block_p95.append(float(np.percentile(lam_u_fit, 95)))
        block_p25.append(float(np.percentile(lam_u_fit, 25)))
        block_p75.append(float(np.percentile(lam_u_fit, 75)))
        block_alpha_norms.append(float(np.linalg.norm(alpha_fit)))
        print(
            f"  B{block['block_idx'] + 1:>2}: mean={block_means[-1]:.3f}  median={block_medians[-1]:.3f}  "
            f"p5..p95=[{block_p5[-1]:.3f}, {block_p95[-1]:.3f}]  ||α||={block_alpha_norms[-1]:.4f}"
        )

    # === Plot 1: per-block λ_u box-style strip with mean line, plus staged c=1 line ===
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    rng = np.random.default_rng(0)

    n_blocks = len(blocks)
    x_idx = np.arange(n_blocks)

    # Staged c=1 reference line
    ax.axhline(1.0, color="#888888", linewidth=1.4, linestyle="--", label="Staged Hawkes: c = 1.000 (degenerate, всех 13 блоков)")

    # Joint per-block: scatter of all 10K lambda_u values per block
    for i, lam_u in enumerate(per_block_lambdas):
        # Subsample for visual clarity; show ~500 points
        sub = rng.choice(lam_u, size=min(500, len(lam_u)), replace=False)
        x_jitter = i + (rng.random(len(sub)) - 0.5) * 0.32
        ax.scatter(x_jitter, sub, s=3, alpha=0.18, color="#2E5EAA")

    # Mean / median lines per block
    for i in range(n_blocks):
        ax.hlines(block_means[i], i - 0.30, i + 0.30, color="#0B3C5D", linewidth=2.2, zorder=5)
        ax.hlines(block_medians[i], i - 0.30, i + 0.30, color="#D2691E", linewidth=1.6, linestyles="--", zorder=5)
        # Annotate mean
        ax.text(i, block_means[i] + 0.05, f"{block_means[i]:.2f}", ha="center", va="bottom", fontsize=8, color="#0B3C5D", fontweight="bold")

    # Legend handles
    legend_handles = [
        plt.Line2D([0], [0], color="#888888", linewidth=1.4, linestyle="--", label="Staged Hawkes: c = 1.000 на всех 13 блоках"),
        plt.Line2D([0], [0], color="#0B3C5D", linewidth=2.2, label="Joint Hawkes: mean λ_u по блоку"),
        plt.Line2D([0], [0], color="#D2691E", linewidth=1.6, linestyle="--", label="Joint Hawkes: median λ_u по блоку"),
        plt.Line2D([0], [0], color="#2E5EAA", marker="o", linestyle="", markersize=4, alpha=0.5, label="Joint Hawkes: per-user λ_u (subsampled to 500)"),
    ]

    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"B{i + 1}" for i in range(n_blocks)])
    ax.set_xlabel("CV block")
    ax.set_ylabel("Множитель перед season_poisson")
    ax.set_title("Per-block распределение множителя на season_poisson: staged vs joint Hawkes (14d train)")
    ax.set_ylim(-0.1, 4.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=legend_handles, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_multiplier_per_block.png", dpi=150)
    plt.close(fig)

    # === Plot 2: histogram of joint λ_u pooled across all 13 blocks ===
    pooled = np.concatenate(per_block_lambdas)
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.hist(pooled, bins=80, color="#2E5EAA", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="#888888", linewidth=1.6, linestyle="--", label="Staged c = 1.000")
    ax.axvline(float(pooled.mean()), color="#0B3C5D", linewidth=2.0, label=f"Joint mean = {pooled.mean():.3f}")
    ax.axvline(float(np.median(pooled)), color="#D2691E", linewidth=1.6, linestyle="--", label=f"Joint median = {np.median(pooled):.3f}")
    ax.set_xlabel("λ_u (joint per-user multiplier на season_poisson)")
    ax.set_ylabel("Количество user-block")
    ax.set_title("Pooled распределение joint λ_u по всем 13 блокам × 10K юзеров")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_multiplier_pooled.png", dpi=150)
    plt.close(fig)

    # Save per-block summary
    summary = pd.DataFrame(
        {
            "block_idx": list(range(n_blocks)),
            "block_label": [f"{blocks[i]['block_start'].date()}..{blocks[i]['train_end'].date()}" for i in range(n_blocks)],
            "staged_c": [1.0] * n_blocks,
            "joint_lambda_mean": block_means,
            "joint_lambda_median": block_medians,
            "joint_lambda_p5": block_p5,
            "joint_lambda_p25": block_p25,
            "joint_lambda_p75": block_p75,
            "joint_lambda_p95": block_p95,
            "joint_alpha_norm": block_alpha_norms,
        }
    )
    summary.to_csv(OUTPUT_DIR / "baseline_multiplier_per_block.csv", index=False)
    print("\nSaved CSV and plots to", OUTPUT_DIR)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

"""Apply chapter-6 frozen Hawkes coefficients to chapter-10 CV blocks.

For each of the 13 three-week blocks (14d train / 7d test) from chapter 10,
compute test NLL of:
  A. Block-fit personalized Gamma-Poisson (chapter 10) — fresh mu_u per block.
  E. Block-fit personalized + FROZEN chapter-6 Hawkes (c, alpha from 207d train) — main comparison.
  D. FROZEN chapter-6 personalized + FROZEN chapter-6 Hawkes — "everything frozen" ceiling.

The main signal is A vs E: same block-fit baseline, but E adds the frozen Hawkes
addition on top. A clean test of "is there a transferable Hawkes signal".
D is shown as a reference point for what happens when even mu_u comes from
the long train (some leakage on in-sample blocks).

The test period of each CV block falls into one of three regimes wrt chapter 6:
  - in-sample for ch.6 (test period <= 2025-08-09): blocks 1..10
  - in-sample-test for ch.6 (2025-08-10 .. 2025-09-30): blocks 11..12
  - extrapolation (after 2025-09-30): block 13

Each row in the output CSV is labeled accordingly; the strip plot uses different
markers for the three regimes.
"""

from __future__ import annotations

import argparse
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

from src.diploma_baselines.data import filter_date_range, load_daily_grid, split_panel_by_date
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_basis_states,
    fit_pooled_additive_multi_kernel_hawkes,
    predict_pooled_additive_multi_kernel_hawkes,
)


# Chapter-6 protocol for the FROZEN models
TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7
CH6_ANALYSIS_START = pd.Timestamp("2025-01-15")
CH6_ANALYSIS_END = pd.Timestamp("2025-09-30")
CH6_TRAIN_RATIO = 0.8
CH6_ALPHA_L2 = 1e-4
CH6_SCALE_L2 = 10.0
CH6_MAX_ITER = 300

# Chapter-10 / 11 / 11b CV protocol
CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
CV_BLOCK_LEN = 21
CV_TRAIN_LEN = 14


def build_blocks(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    blocks: list[dict] = []
    cursor = start
    idx = 0
    while True:
        block_start = cursor
        block_end = cursor + pd.Timedelta(days=CV_BLOCK_LEN - 1)
        if block_end > end:
            break
        train_end = block_start + pd.Timedelta(days=CV_TRAIN_LEN - 1)
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
    return blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply chapter-6 frozen models to chapter-10 CV blocks")
    parser.add_argument(
        "--data-path",
        default="data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        help="Path to daily grid csv",
    )
    parser.add_argument(
        "--ch10-cv-csv",
        default="diploma/reports/blockwise_cv/cv_results.csv",
        help="Per-block NLL csv from chapter 10",
    )
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/blockwise_cv_frozen_ch6",
        help="Directory for output artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(args.data_path, value_cols=cols)
    print(f"  loaded {len(full_df):,} rows")

    # === Step 1. Fit chapter-6 personalized + Hawkes ===
    print("\n=== Fitting chapter-6 frozen models ===")
    t0 = time.time()
    analysis_df_ch6 = filter_date_range(full_df, start_date=CH6_ANALYSIS_START, end_date=CH6_ANALYSIS_END)
    split_ch6 = split_panel_by_date(analysis_df_ch6, train_ratio=CH6_TRAIN_RATIO)
    print(f"  ch.6 split_date = {split_ch6.split_date.date()}, train rows = {len(split_ch6.train):,}")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    train_daily_mean_ch6 = split_ch6.train.groupby("event_date")[TARGET_COL].mean().sort_index()

    rolling_seasonal_ch6 = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean_ch6,
        daily_mean_full,
    )

    base_train_ch6 = rolling_seasonal_ch6.predict_for_dates(split_ch6.train["event_date"]).to_numpy(dtype=float)
    scaler_ch6 = PersonalizedGammaPoissonScaler().fit(
        split_ch6.train["user_id"].to_numpy(),
        split_ch6.train[TARGET_COL].to_numpy(),
        base_train_ch6,
    )
    print(f"  ch.6 EB: alpha={scaler_ch6.alpha_:.4f}, beta={scaler_ch6.beta_:.4f}")

    # Predict personalized for ALL user-days in the union of (ch.6 analysis range) and (CV range).
    # CV extends to Oct-31, ch.6 ends Sep-30, so combined range is Jan-15 .. Oct-31.
    union_df = filter_date_range(full_df, start_date=CV_GLOBAL_START, end_date=CV_GLOBAL_END).copy()
    base_union = rolling_seasonal_ch6.predict_for_dates(union_df["event_date"]).to_numpy(dtype=float)
    pers_union = scaler_ch6.predict(union_df["user_id"].to_numpy(), base_union, method="posterior_mean")
    union_df["frozen_ch6_personalized"] = pers_union

    # === Step 2. Fit chapter-6 Hawkes ===
    print("  fitting Hawkes on ch.6 train...")
    beta = np.log(2.0) / np.asarray(HALF_LIVES, dtype=float)

    # Build states for full data once
    user_states: dict[int, dict] = {}
    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        user_states[int(user_id)] = {
            "states_full": states_full,
            "dates": full_user["event_date"].to_numpy(dtype="datetime64[ns]"),
        }

    # Build train state/y/base blocks for ch.6 fit
    ch6_train_start64 = np.datetime64(split_ch6.train["event_date"].min())
    ch6_train_end64 = np.datetime64(split_ch6.train["event_date"].max())

    pers_train_ch6 = scaler_ch6.predict(split_ch6.train["user_id"].to_numpy(), base_train_ch6, method="posterior_mean")
    pers_train_by_user: dict[int, np.ndarray] = {}
    for uid, idx in split_ch6.train.groupby("user_id", sort=False).indices.items():
        pers_train_by_user[int(uid)] = pers_train_ch6[idx]

    train_state_blocks: list[np.ndarray] = []
    train_y_blocks: list[np.ndarray] = []
    train_base_blocks: list[np.ndarray] = []
    for user_id, info in user_states.items():
        dates = info["dates"]
        in_train = (dates >= ch6_train_start64) & (dates <= ch6_train_end64)
        if not in_train.any():
            continue
        base_t = pers_train_by_user.get(int(user_id))
        if base_t is None:
            continue
        # Locate y for those days from full_df
        full_user_y = full_df.loc[full_df["user_id"] == user_id, TARGET_COL].to_numpy(dtype=float)
        train_state_blocks.append(info["states_full"][in_train])
        train_y_blocks.append(full_user_y[in_train])
        train_base_blocks.append(base_t)

    hawkes_ch6 = fit_pooled_additive_multi_kernel_hawkes(
        state_blocks=train_state_blocks,
        y_blocks=train_y_blocks,
        base_blocks=train_base_blocks,
        half_lives=HALF_LIVES,
        feature_names=HAWKES_FEATURES,
        alpha_l2=CH6_ALPHA_L2,
        learn_base_scale=True,
        scale_l2=CH6_SCALE_L2,
        scale_init=1.0,
        max_iter=CH6_MAX_ITER,
    )
    alpha_norm = float(np.linalg.norm(hawkes_ch6.alpha))
    print(f"  ch.6 Hawkes: c={hawkes_ch6.base_scale:.4f}, ||alpha||_2={alpha_norm:.4f}")

    # === Step 3. Predict frozen ch.6 Hawkes for the CV union period ===
    print("  predicting frozen ch.6 Hawkes for CV union period...")
    union_pred_hawkes = np.zeros(len(union_df), dtype=float)
    union_user_groups = union_df.groupby("user_id", sort=False).indices
    union_dates = union_df["event_date"].to_numpy(dtype="datetime64[ns]")

    for user_id, idx in union_user_groups.items():
        info = user_states[int(user_id)]
        full_dates = info["dates"]
        # Match each row of union for this user with the corresponding full-data row
        wanted_dates = union_dates[idx]
        # Build a map full_date -> row
        full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}
        rows_in_full = np.array([full_to_idx[pd.Timestamp(d).normalize()] for d in wanted_dates], dtype=int)
        states_for_block = info["states_full"][rows_in_full]
        base_for_block = pers_union[idx]
        lam, _ = predict_pooled_additive_multi_kernel_hawkes(
            hawkes_ch6,
            states=states_for_block,
            base_lambda=base_for_block,
        )
        union_pred_hawkes[idx] = lam
    union_df["frozen_ch6_hawkes"] = union_pred_hawkes
    print(f"  done in {time.time() - t0:.1f}s")

    # === Step 4. For each CV block, compute test NLL of frozen ch.6 settings ===
    blocks = build_blocks(CV_GLOBAL_START, CV_GLOBAL_END)

    # Determine block regime wrt chapter 6
    def regime(test_start: pd.Timestamp, test_end: pd.Timestamp) -> str:
        ch6_train_end_ts = pd.Timestamp(split_ch6.split_date)
        ch6_test_end_ts = CH6_ANALYSIS_END
        if test_end <= ch6_train_end_ts:
            return "in_ch6_train"
        if test_start > ch6_train_end_ts and test_end <= ch6_test_end_ts:
            return "in_ch6_test"
        if test_start > ch6_test_end_ts:
            return "extrapolation"
        return "boundary"

    # Pre-compute daily mean for rolling-seasonal block fits
    daily_mean_full_target = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    rows: list[dict] = []
    print("\n=== Per-block fits ===")
    for block in blocks:
        block_idx = block["block_idx"]
        block_start = block["block_start"]
        block_end = block["block_end"]
        train_end = block["train_end"]
        test_start = block["test_start"]
        test_end = block["test_end"]

        test_mask = (
            (union_df["event_date"] >= test_start) & (union_df["event_date"] <= test_end)
        )
        block_test = union_df.loc[test_mask].copy()
        y_true = block_test[TARGET_COL].to_numpy(dtype=float)

        # --- Setting D: frozen ch.6 personalized + frozen ch.6 Hawkes ---
        nll_d_pers = evaluate_count_forecast(
            y_true, block_test["frozen_ch6_personalized"].to_numpy(dtype=float)
        )["mean_poisson_nll"]
        nll_d_hawkes = evaluate_count_forecast(
            y_true, block_test["frozen_ch6_hawkes"].to_numpy(dtype=float)
        )["mean_poisson_nll"]

        # --- Setting A: block-fit personalized (re-fit per block, same protocol as chapter 10) ---
        block_train_mask = (
            (full_df["event_date"] >= block_start) & (full_df["event_date"] <= train_end)
        )
        block_test_mask = (
            (full_df["event_date"] >= test_start) & (full_df["event_date"] <= test_end)
        )
        block_train_df = full_df.loc[block_train_mask].copy()
        block_test_df = full_df.loc[block_test_mask].copy()

        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs_block = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean,
            daily_mean_full_target,
        )
        base_train_block = rs_block.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)
        base_test_block = rs_block.predict_for_dates(block_test_df["event_date"]).to_numpy(dtype=float)
        scaler_block = PersonalizedGammaPoissonScaler().fit(
            block_train_df["user_id"].to_numpy(),
            block_train_df[TARGET_COL].to_numpy(),
            base_train_block,
        )
        pers_test_block = scaler_block.predict(
            block_test_df["user_id"].to_numpy(),
            base_test_block,
            method="posterior_mean",
        )
        nll_a = evaluate_count_forecast(
            block_test_df[TARGET_COL].to_numpy(dtype=float), pers_test_block
        )["mean_poisson_nll"]

        # --- Setting E: block-fit personalized + frozen ch.6 Hawkes ---
        # For each test row we already have block-fit personalized prediction (pers_test_block).
        # Apply frozen ch.6 Hawkes on top: lam_u(t) = c * mu_u(t) + alpha @ states_u(t)
        block_test_dates = block_test_df["event_date"].to_numpy(dtype="datetime64[ns]")
        e_pred = np.zeros(len(block_test_df), dtype=float)
        for uid, idx_in_block in block_test_df.groupby("user_id", sort=False).indices.items():
            info = user_states[int(uid)]
            full_dates = info["dates"]
            wanted_dates = block_test_dates[idx_in_block]
            full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}
            rows_in_full = np.array([full_to_idx[pd.Timestamp(d).normalize()] for d in wanted_dates], dtype=int)
            states_for_block = info["states_full"][rows_in_full]
            base_for_block = pers_test_block[idx_in_block]
            lam, _ = predict_pooled_additive_multi_kernel_hawkes(
                hawkes_ch6,
                states=states_for_block,
                base_lambda=base_for_block,
            )
            e_pred[idx_in_block] = lam
        nll_e = evaluate_count_forecast(
            block_test_df[TARGET_COL].to_numpy(dtype=float), e_pred
        )["mean_poisson_nll"]

        block_regime = regime(test_start, test_end)
        print(
            f"  block {block_idx + 1:>2}/{len(blocks)} {block_start.date()}..{block_end.date()} "
            f"[{block_regime:<13}] "
            f"A={nll_a:.4f}  E={nll_e:.4f}  D={nll_d_hawkes:.4f}  ΔE-A={nll_e - nll_a:+.4f}"
        )

        rows.append(
            {
                "block_idx": block_idx,
                "block_label": f"{block_start.date()}..{block_end.date()}",
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "test_rows": int(len(block_test)),
                "regime_vs_ch6": block_regime,
                "block_fit_personalized_nll": float(nll_a),
                "block_fit_pers_plus_frozen_hawkes_nll": float(nll_e),
                "frozen_ch6_personalized_nll": float(nll_d_pers),
                "frozen_ch6_hawkes_nll": float(nll_d_hawkes),
            }
        )

    frozen_df = pd.DataFrame(rows)

    merged = frozen_df
    merged.to_csv(output_dir / "frozen_ch6_per_block.csv", index=False)

    # === Step 6. Build strip plot ===
    settings = [
        ("A: Block-fit Personalized", "block_fit_personalized_nll", "#2E5EAA"),
        ("E: Block-fit Pers + Frozen ch.6 Hawkes", "block_fit_pers_plus_frozen_hawkes_nll", "#2E8B57"),
        ("D: Frozen ch.6 Pers + Frozen ch.6 Hawkes", "frozen_ch6_hawkes_nll", "#D2691E"),
    ]

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    summary: dict[str, dict[str, float]] = {}

    regime_marker = {"in_ch6_train": "o", "in_ch6_test": "s", "boundary": "P", "extrapolation": "X"}

    for i, (label, col, color) in enumerate(settings):
        vals = merged[col].to_numpy(dtype=float)
        regimes = merged["regime_vs_ch6"].to_numpy()
        x_jitter = i + (rng.random(len(vals)) - 0.5) * 0.32
        for x, y, r in zip(x_jitter, vals, regimes):
            marker = regime_marker.get(r, "o")
            ax.scatter([x], [y], color=color, alpha=0.75, s=55, edgecolors="white", linewidths=0.7, marker=marker, zorder=3)
        mean_val = float(np.nanmean(vals))
        median_val = float(np.nanmedian(vals))
        ax.hlines(mean_val, i - 0.30, i + 0.30, color="#0B3C5D", linewidth=2.4, zorder=4, label="Mean" if i == 0 else None)
        ax.hlines(median_val, i - 0.30, i + 0.30, color="#D2691E", linewidth=1.8, linestyles="--", zorder=4, label="Median" if i == 0 else None)
        ax.text(i, mean_val, f"{mean_val:.4f}", ha="center", va="bottom", fontsize=9, color="#0B3C5D", fontweight="bold")
        summary[label] = {
            "mean_nll": mean_val,
            "median_nll": median_val,
            "std_nll": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(np.sum(~np.isnan(vals))),
        }

    # Marker legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="#888888", linestyle="", label="block in ch.6 train"),
        plt.Line2D([0], [0], marker="s", color="#888888", linestyle="", label="block in ch.6 test"),
        plt.Line2D([0], [0], marker="X", color="#888888", linestyle="", label="block extrapolation"),
        plt.Line2D([0], [0], color="#0B3C5D", linewidth=2.4, label="Mean"),
        plt.Line2D([0], [0], color="#D2691E", linewidth=1.8, linestyle="--", label="Median"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)

    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels([lbl.replace(": ", "\n") for lbl in [s[0] for s in settings]], fontsize=9)
    ax.set_ylabel("Test NLL per user-day (lower is better)")
    ax.set_title("Per-block test NLL: block-fit vs frozen ch.6 models on 13 three-week blocks")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "frozen_ch6_strip_plot.png", dpi=150)
    plt.close(fig)

    # === Step 7. Per-block delta plot: E - A (frozen Hawkes contribution on honest block-fit baseline) ===
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    delta_ea = (
        merged["block_fit_pers_plus_frozen_hawkes_nll"] - merged["block_fit_personalized_nll"]
    ).to_numpy(dtype=float)
    regime_color = {
        "in_ch6_train": "#888888",
        "boundary": "#A0522D",
        "in_ch6_test": "#0B3C5D",
        "extrapolation": "#2E8B57",
    }
    bar_colors = [regime_color.get(r, "#888888") for r in merged["regime_vs_ch6"]]
    x_idx = np.arange(len(merged))
    ax.bar(x_idx, delta_ea, color=bar_colors, edgecolor="white")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"B{i + 1}\n{lbl}" for i, lbl in enumerate(merged["block_label"])], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Δ NLL/n: (block-fit pers + frozen Hawkes) − block-fit pers")
    ax.set_title("Чистый вклад frozen ch.6 Hawkes поверх честного block-fit baseline (отрицательное = улучшение)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=regime_color["in_ch6_train"], label="block in ch.6 train (in-sample)"),
        plt.Rectangle((0, 0), 1, 1, color=regime_color["boundary"], label="boundary block"),
        plt.Rectangle((0, 0), 1, 1, color=regime_color["in_ch6_test"], label="block in ch.6 test (out-of-sample)"),
        plt.Rectangle((0, 0), 1, 1, color=regime_color["extrapolation"], label="block extrapolation (Oct)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "frozen_ch6_hawkes_contribution.png", dpi=150)
    plt.close(fig)

    summary_out = {
        "ch6_protocol": {
            "train_window": [str(CH6_ANALYSIS_START.date()), str(split_ch6.split_date.date())],
            "n_train_rows": int(len(split_ch6.train)),
            "ch6_eb_alpha": float(scaler_ch6.alpha_),
            "ch6_eb_beta": float(scaler_ch6.beta_),
            "ch6_hawkes_c": float(hawkes_ch6.base_scale),
            "ch6_hawkes_alpha_norm": float(alpha_norm),
        },
        "n_cv_blocks": int(len(merged)),
        "settings_summary": summary,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    print("\nPer-setting summary:")
    for label, m in summary.items():
        print(f"  {label:<40s}  mean={m['mean_nll']:.4f}  median={m['median_nll']:.4f}  std={m['std_nll']:.4f}  (n={m['n']})")


if __name__ == "__main__":
    main()

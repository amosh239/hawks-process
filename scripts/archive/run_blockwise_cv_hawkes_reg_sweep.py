"""Blockwise CV sweep over Hawkes regularization (alpha_l2, scale_l2).

Same 21-day blocks as chapter 10 (14d train / 7d test).
For each block and each (alpha_l2, scale_l2) setting, fit pooled scaled-baseline Hawkes
and record:
  - test NLL per user-day,
  - learned base scale c,
  - ||alpha||_2,
  - whether the fit landed at the trivial (c=1, alpha=0) point.

Personalized baseline is also re-fit per block as the reference. The artifacts
support chapter 11's diagnostic of why Hawkes degenerates on short training windows.
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


HAWKES_FEATURES = tuple(FEATURE_NAMES)
TARGET_COL = "to_ord"
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7
GLOBAL_START = pd.Timestamp("2025-01-15")
GLOBAL_END = pd.Timestamp("2025-10-31")
DEFAULT_BLOCK_LEN = 21
DEFAULT_TRAIN_LEN = 14
SCALE_INIT = 1.0
MAX_ITER = 300


# (label, alpha_l2, scale_l2)
# (label, alpha_l2, scale_l2, warm_start_alpha)
# warm_start_alpha=True means seed optimizer with chapter-6's long-train solution
SETTINGS = [
    ("A: default (1e-4, 10.0)", 1e-4, 10.0, False),
    ("B: alpha_l2=0, scale_l2=10.0", 0.0, 10.0, False),
    ("C: alpha_l2=1e-4, scale_l2=0.1", 1e-4, 0.1, False),
    ("D: alpha_l2=1e-4, scale_l2=0.0", 1e-4, 0.0, False),
    ("E: no regularization", 0.0, 0.0, False),
    ("F: warm-start from ch.6, no reg", 0.0, 0.0, True),
]


# Long-train Hawkes solution from chapter 6 (loaded from chapter-6 alpha_table.csv).
# Indexed by (feature_idx, half_life_idx); features = ["searches","cat_to_cart","cat_to_ord","to_cart","to_ord"], half_lives = [1, 3].
LONG_TRAIN_ALPHA = np.array(
    [
        [0.003, 0.000],   # searches
        [0.001, 0.000],   # cat_to_cart
        [0.000, 0.001],   # cat_to_ord
        [0.004, 0.000],   # to_cart
        [0.000, 0.015],   # to_ord
    ],
    dtype=float,
)
LONG_TRAIN_SCALE = 0.8262


def build_blocks(start: pd.Timestamp, end: pd.Timestamp, block_len: int, train_len: int) -> list[dict]:
    blocks: list[dict] = []
    cursor = start
    idx = 0
    while True:
        block_start = cursor
        block_end = cursor + pd.Timedelta(days=block_len - 1)
        if block_end > end:
            break
        train_end = block_start + pd.Timedelta(days=train_len - 1)
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
    return blocks


def precompute_user_states(full_df: pd.DataFrame) -> dict[int, dict]:
    """Build Hawkes states once per user; reused across blocks/settings."""
    beta = np.log(2.0) / np.asarray(HAWKES_HALF_LIVES_ARR := HALF_LIVES, dtype=float)
    user_states: dict[int, dict] = {}
    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        user_states[int(user_id)] = {
            "states_full": states_full,
            "dates": full_user["event_date"].to_numpy(dtype="datetime64[ns]"),
            "y_full": full_user[TARGET_COL].to_numpy(dtype=float),
        }
    return user_states


def fit_personalized_for_block(
    full_df: pd.DataFrame,
    block_df: pd.DataFrame,
    train_end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Fit RollingSeasonal + Gamma-Poisson on this block's train, predict on entire block.

    Returns (train_pred_arr, test_pred_arr, train_pred_by_user, test_pred_by_user).
    """
    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    train_df = block_df[block_df["event_date"] <= train_end].copy()
    test_df = block_df[block_df["event_date"] > train_end].copy()
    train_daily_mean = train_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    rolling_seasonal = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean,
        daily_mean_full,
    )
    base_train = rolling_seasonal.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    base_test = rolling_seasonal.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)

    scaler = PersonalizedGammaPoissonScaler().fit(
        train_df["user_id"].to_numpy(),
        train_df[TARGET_COL].to_numpy(),
        base_train,
    )
    train_pred = scaler.predict(train_df["user_id"].to_numpy(), base_train, method="posterior_mean")
    test_pred = scaler.predict(test_df["user_id"].to_numpy(), base_test, method="posterior_mean")

    train_pred_by_user: dict[int, np.ndarray] = {}
    test_pred_by_user: dict[int, np.ndarray] = {}
    for uid, idx in train_df.groupby("user_id", sort=False).indices.items():
        train_pred_by_user[int(uid)] = train_pred[idx]
    for uid, idx in test_df.groupby("user_id", sort=False).indices.items():
        test_pred_by_user[int(uid)] = test_pred[idx]

    return train_pred, test_pred, train_pred_by_user, test_pred_by_user, train_df, test_df


def fit_and_eval_hawkes(
    user_states: dict[int, dict],
    train_pred_by_user: dict[int, np.ndarray],
    test_pred_by_user: dict[int, np.ndarray],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    block_start: pd.Timestamp,
    train_end: pd.Timestamp,
    block_end: pd.Timestamp,
    alpha_l2: float,
    scale_l2: float,
    warm_start: bool = False,
) -> dict:
    block_start64 = np.datetime64(block_start)
    train_end64 = np.datetime64(train_end)
    block_end64 = np.datetime64(block_end)

    train_state_blocks: list[np.ndarray] = []
    train_y_blocks: list[np.ndarray] = []
    train_base_blocks: list[np.ndarray] = []
    test_state_blocks: list[np.ndarray] = []
    test_y_blocks: list[np.ndarray] = []
    test_base_blocks: list[np.ndarray] = []

    train_user_ids = set(train_df["user_id"].astype(int).unique())
    test_user_ids = set(test_df["user_id"].astype(int).unique())

    for user_id, info in user_states.items():
        dates = info["dates"]
        states_full = info["states_full"]
        y_full = info["y_full"]
        train_mask = (dates >= block_start64) & (dates <= train_end64)
        test_mask = (dates > train_end64) & (dates <= block_end64)

        if int(user_id) in train_user_ids and train_mask.any():
            base_t = train_pred_by_user.get(int(user_id))
            if base_t is not None:
                train_state_blocks.append(states_full[train_mask])
                train_y_blocks.append(y_full[train_mask])
                train_base_blocks.append(base_t)

        if int(user_id) in test_user_ids and test_mask.any():
            base_t = test_pred_by_user.get(int(user_id))
            if base_t is not None:
                test_state_blocks.append(states_full[test_mask])
                test_y_blocks.append(y_full[test_mask])
                test_base_blocks.append(base_t)

    if warm_start:
        alpha_init = LONG_TRAIN_ALPHA.reshape(-1)
        scale_init = LONG_TRAIN_SCALE
    else:
        alpha_init = None
        scale_init = SCALE_INIT
    hawkes = fit_pooled_additive_multi_kernel_hawkes(
        state_blocks=train_state_blocks,
        y_blocks=train_y_blocks,
        base_blocks=train_base_blocks,
        half_lives=HALF_LIVES,
        feature_names=HAWKES_FEATURES,
        alpha_l2=float(alpha_l2),
        learn_base_scale=True,
        scale_l2=float(scale_l2),
        scale_init=float(scale_init),
        alpha_init=alpha_init,
        max_iter=MAX_ITER,
    )

    test_preds: list[np.ndarray] = []
    for states, base in zip(test_state_blocks, test_base_blocks):
        lam, _ = predict_pooled_additive_multi_kernel_hawkes(
            hawkes,
            states=states,
            base_lambda=base,
        )
        test_preds.append(lam)
    pred_concat = np.concatenate(test_preds)
    target_concat = np.concatenate(test_y_blocks)
    metrics = evaluate_count_forecast(target_concat, pred_concat)

    alpha_norm = float(np.linalg.norm(hawkes.alpha))
    is_degenerate = (abs(hawkes.base_scale - 1.0) < 1e-3) and (alpha_norm < 1e-4)

    return {
        "test_nll": float(metrics["mean_poisson_nll"]),
        "learned_base_scale": float(hawkes.base_scale),
        "alpha_l2_norm": alpha_norm,
        "fit_success": bool(hawkes.success),
        "is_degenerate": bool(is_degenerate),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blockwise CV sweep over Hawkes regularization")
    parser.add_argument(
        "--data-path",
        default="data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        help="Path to daily grid csv",
    )
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/blockwise_cv_hawkes_reg",
        help="Directory for sweep artifacts",
    )
    parser.add_argument(
        "--block-len",
        type=int,
        default=DEFAULT_BLOCK_LEN,
        help="Total block length in days (train + test)",
    )
    parser.add_argument(
        "--train-len",
        type=int,
        default=DEFAULT_TRAIN_LEN,
        help="Train portion length in days within each block",
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

    print("Precomputing Hawkes states per user (one-shot)...")
    t0 = time.time()
    user_states = precompute_user_states(full_df)
    print(f"  done in {time.time() - t0:.1f}s")

    blocks = build_blocks(GLOBAL_START, GLOBAL_END, args.block_len, args.train_len)
    test_len = args.block_len - args.train_len
    print(f"  built {len(blocks)} blocks: {args.train_len}d train / {test_len}d test each")

    rows: list[dict] = []
    baseline_rows: list[dict] = []

    for block in blocks:
        block_idx = block["block_idx"]
        block_start = block["block_start"]
        block_end = block["block_end"]
        train_end = block["train_end"]
        block_label = f"{block_start.date()}..{block_end.date()}"
        print(f"\n[Block {block_idx + 1}/{len(blocks)}] {block_label}")

        block_df = filter_date_range(full_df, start_date=block_start, end_date=block_end)
        if block_df.empty:
            continue

        (
            _train_pred_arr,
            _test_pred_arr,
            train_pred_by_user,
            test_pred_by_user,
            train_df,
            test_df,
        ) = fit_personalized_for_block(full_df, block_df, train_end)

        baseline_metrics = evaluate_count_forecast(
            test_df[TARGET_COL].to_numpy(),
            np.concatenate([test_pred_by_user[uid] for uid in test_df["user_id"].astype(int).unique()]),
        )
        baseline_nll = float(baseline_metrics["mean_poisson_nll"])
        baseline_rows.append({"block_idx": block_idx, "block_label": block_label, "personalized_nll": baseline_nll})
        print(f"  personalized baseline NLL = {baseline_nll:.4f}")

        for label, alpha_l2, scale_l2, warm_start in SETTINGS:
            t0 = time.time()
            res = fit_and_eval_hawkes(
                user_states=user_states,
                train_pred_by_user=train_pred_by_user,
                test_pred_by_user=test_pred_by_user,
                train_df=train_df,
                test_df=test_df,
                block_start=block_start,
                train_end=train_end,
                block_end=block_end,
                alpha_l2=alpha_l2,
                scale_l2=scale_l2,
                warm_start=warm_start,
            )
            elapsed = time.time() - t0
            row = {
                "block_idx": block_idx,
                "block_label": block_label,
                "setting_label": label,
                "alpha_l2": alpha_l2,
                "scale_l2": scale_l2,
                "personalized_nll": baseline_nll,
                **res,
                "delta_vs_personalized_nll": res["test_nll"] - baseline_nll,
                "fit_seconds": elapsed,
            }
            rows.append(row)
            print(
                f"  [{label}]"
                f" NLL={res['test_nll']:.4f}"
                f" Δvs base={row['delta_vs_personalized_nll']:+.4f}"
                f" c={res['learned_base_scale']:.4f}"
                f" ||α||={res['alpha_l2_norm']:.4f}"
                f" {'(DEGENERATE)' if res['is_degenerate'] else ''}"
                f" ({elapsed:.1f}s)"
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "cv_reg_sweep_results.csv", index=False)

    # Aggregate
    agg = (
        df.groupby("setting_label")
        .agg(
            n=("test_nll", "count"),
            mean_nll=("test_nll", "mean"),
            median_nll=("test_nll", "median"),
            mean_delta=("delta_vs_personalized_nll", "mean"),
            mean_c=("learned_base_scale", "mean"),
            median_c=("learned_base_scale", "median"),
            mean_alpha_norm=("alpha_l2_norm", "mean"),
            median_alpha_norm=("alpha_l2_norm", "median"),
            degenerate_share=("is_degenerate", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(output_dir / "cv_reg_sweep_aggregate.csv", index=False)

    summary = {
        "n_blocks": int(df["block_idx"].nunique()),
        "block_len_days": int(args.block_len),
        "train_len_days": int(args.train_len),
        "test_len_days": int(args.block_len - args.train_len),
        "settings": [
            {
                "label": label,
                "alpha_l2": alpha_l2,
                "scale_l2": scale_l2,
                "warm_start": warm_start,
                **agg.loc[agg["setting_label"] == label].iloc[0].drop("setting_label").to_dict(),
            }
            for label, alpha_l2, scale_l2, warm_start in SETTINGS
        ],
        "baseline_mean_nll": float(np.mean([r["personalized_nll"] for r in baseline_rows])),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Plot 1: Strip plot of NLL per setting
    settings_labels = [s[0] for s in SETTINGS]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    for i, label in enumerate(settings_labels):
        vals = df.loc[df["setting_label"] == label, "test_nll"].to_numpy(dtype=float)
        x_jitter = i + (rng.random(len(vals)) - 0.5) * 0.32
        ax.scatter(x_jitter, vals, color="#0B3C5D", alpha=0.65, s=42, edgecolors="white", linewidths=0.6)
        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        ax.hlines(mean_val, i - 0.28, i + 0.28, color="#0B3C5D", linewidth=2.4, label="Mean" if i == 0 else None, zorder=4)
        ax.hlines(median_val, i - 0.28, i + 0.28, color="#D2691E", linewidth=1.8, linestyles="--", label="Median" if i == 0 else None, zorder=4)
        ax.text(i, mean_val, f"{mean_val:.4f}", ha="center", va="bottom", fontsize=9, color="#0B3C5D", fontweight="bold")

    baseline_mean = float(np.mean([r["personalized_nll"] for r in baseline_rows]))
    ax.axhline(baseline_mean, color="#888888", linestyle="--", linewidth=1.2,
               label=f"Personalized baseline mean = {baseline_mean:.4f}")
    ax.set_xticks(range(len(settings_labels)))
    ax.set_xticklabels([lbl.replace(": ", "\n") for lbl in settings_labels], fontsize=9)
    ax.set_ylabel("Test NLL per user-day (lower is better)")
    ax.set_title(
        f"Blockwise CV ({args.train_len}d train / {test_len}d test, {len(blocks)} blocks): "
        "Hawkes test NLL by regularization setting"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "cv_reg_sweep_nll.png", dpi=150)
    plt.close(fig)

    # Plot 2: Strip plot of fitted c and ||alpha||
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    ax = axes[0]
    for i, label in enumerate(settings_labels):
        vals = df.loc[df["setting_label"] == label, "learned_base_scale"].to_numpy(dtype=float)
        x_jitter = i + (rng.random(len(vals)) - 0.5) * 0.32
        ax.scatter(x_jitter, vals, color="#2E5EAA", alpha=0.65, s=42, edgecolors="white", linewidths=0.6)
        mean_val = float(np.mean(vals))
        ax.hlines(mean_val, i - 0.28, i + 0.28, color="#0B3C5D", linewidth=2.4, zorder=4)
        ax.text(i, mean_val, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=9, color="#0B3C5D", fontweight="bold")
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.text(len(settings_labels) - 0.5, 1.0, "c = 1 (degenerate)", ha="right", va="bottom", fontsize=9, color="#888888")
    ax.set_xticks(range(len(settings_labels)))
    ax.set_xticklabels([lbl.split(":")[0] for lbl in settings_labels], fontsize=9)
    ax.set_ylabel("Fitted base scale c")
    ax.set_title("Fitted c per block and per setting")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    for i, label in enumerate(settings_labels):
        vals = df.loc[df["setting_label"] == label, "alpha_l2_norm"].to_numpy(dtype=float)
        x_jitter = i + (rng.random(len(vals)) - 0.5) * 0.32
        ax.scatter(x_jitter, vals, color="#D2691E", alpha=0.65, s=42, edgecolors="white", linewidths=0.6)
        mean_val = float(np.mean(vals))
        ax.hlines(mean_val, i - 0.28, i + 0.28, color="#A0522D", linewidth=2.4, zorder=4)
        ax.text(i, mean_val, f"{mean_val:.4f}", ha="center", va="bottom", fontsize=9, color="#A0522D", fontweight="bold")
    ax.set_xticks(range(len(settings_labels)))
    ax.set_xticklabels([lbl.split(":")[0] for lbl in settings_labels], fontsize=9)
    ax.set_ylabel("||α||_2")
    ax.set_title("||α||_2 per block and per setting")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "cv_reg_sweep_structure.png", dpi=150)
    plt.close(fig)

    print("\nAggregate per setting:")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()

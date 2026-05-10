"""Blockwise 3-week cross-validation across all main models.

We split [2025-01-15 .. 2025-10-31] into non-overlapping 21-day blocks.
Within each block: first 14 days are train, last 7 days are test.
For every block, we fit and evaluate each of the 6 models from the ladder
(plus experimental GBDT) and collect their per-observation test NLL.
"""

from __future__ import annotations

import argparse
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diploma_baselines.data import filter_date_range, load_daily_grid, split_panel_by_date
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    DEFAULT_HALF_LIVES,
    FEATURE_NAMES,
    GlobalPoissonModel,
    GlobalRollingMeanPoissonModel,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    UserStatesCache,
    build_user_states_cache,
    fit_pooled_additive_multi_kernel_hawkes,
    predict_pooled_additive_multi_kernel_hawkes,
)
from src.diploma_experimental.gbdt import SOURCE_FEATURES, build_feature_tables, fit_global_poisson_gbdt


BLOCK_LEN = 21
TRAIN_LEN = 14
GLOBAL_START = pd.Timestamp("2025-01-15")
GLOBAL_END = pd.Timestamp("2025-10-31")

ROLLING_WINDOW = 7
HAWKES_HALF_LIVES = (1.0, 3.0)
HAWKES_FEATURES = tuple(FEATURE_NAMES)
TARGET_COL = "to_ord"

GBDT_PARAMS = {
    "seed": 42,
    "max_depth": 5,
    "learning_rate": 0.05,
    "max_iter": 200,
    "min_samples_leaf": 40,
}


def build_blocks(start: pd.Timestamp, end: pd.Timestamp, block_len: int = BLOCK_LEN) -> list[dict]:
    blocks: list[dict] = []
    cursor = start
    idx = 0
    while True:
        block_start = cursor
        block_end = cursor + pd.Timedelta(days=block_len - 1)
        if block_end > end:
            break
        train_end = block_start + pd.Timedelta(days=TRAIN_LEN - 1)
        test_start = train_end + pd.Timedelta(days=1)
        blocks.append(
            {
                "block_idx": idx,
                "block_start": block_start,
                "block_end": block_end,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": block_end,
            }
        )
        idx += 1
        cursor = block_end + pd.Timedelta(days=1)
    return blocks


def evaluate_global_poisson(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    model = GlobalPoissonModel().fit(train_df[TARGET_COL].to_numpy())
    pred = model.predict(len(test_df))
    return evaluate_count_forecast(test_df[TARGET_COL].to_numpy(), pred)["mean_poisson_nll"]


def evaluate_rolling(daily_mean_full: pd.Series, test_df: pd.DataFrame) -> float:
    model = GlobalRollingMeanPoissonModel(window_size=ROLLING_WINDOW, min_periods=1).fit(daily_mean_full)
    pred = model.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)
    return evaluate_count_forecast(test_df[TARGET_COL].to_numpy(), pred)["mean_poisson_nll"]


def evaluate_rolling_seasonal(
    train_daily_mean: pd.Series,
    daily_mean_full: pd.Series,
    test_df: pd.DataFrame,
) -> tuple[float, np.ndarray]:
    model = GlobalRollingSeasonalPoissonModel(window_size=ROLLING_WINDOW, min_periods=1).fit(
        train_daily_mean,
        daily_mean_full,
    )
    pred = model.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)
    nll = evaluate_count_forecast(test_df[TARGET_COL].to_numpy(), pred)["mean_poisson_nll"]
    return nll, model


def evaluate_personalized(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_model: GlobalRollingSeasonalPoissonModel,
) -> tuple[float, np.ndarray, np.ndarray, PersonalizedGammaPoissonScaler]:
    base_train = base_model.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    base_test = base_model.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)
    scaler = PersonalizedGammaPoissonScaler().fit(
        train_df["user_id"].to_numpy(),
        train_df[TARGET_COL].to_numpy(),
        base_train,
    )
    train_pred = scaler.predict(train_df["user_id"].to_numpy(), base_train, method="posterior_mean")
    test_pred = scaler.predict(test_df["user_id"].to_numpy(), base_test, method="posterior_mean")
    nll = evaluate_count_forecast(test_df[TARGET_COL].to_numpy(), test_pred)["mean_poisson_nll"]
    return nll, train_pred, test_pred, scaler


def evaluate_hawkes(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    block_start: pd.Timestamp,
    block_end: pd.Timestamp,
    train_end: pd.Timestamp,
    train_personalized_pred: np.ndarray,
    test_personalized_pred: np.ndarray,
    cache: UserStatesCache,
) -> float:
    """Fit pooled scaled-baseline Hawkes on this block's train, predict on its test."""
    states_train_flat = cache.gather_for(train_df)
    states_test_flat = cache.gather_for(test_df)
    y_train_flat = train_df[TARGET_COL].to_numpy(dtype=float)
    y_test_flat = test_df[TARGET_COL].to_numpy(dtype=float)
    train_pred_array = np.asarray(train_personalized_pred, dtype=float)
    test_pred_array = np.asarray(test_personalized_pred, dtype=float)

    train_state_blocks: list[np.ndarray] = []
    train_y_blocks: list[np.ndarray] = []
    train_base_blocks: list[np.ndarray] = []
    test_state_blocks: list[np.ndarray] = []
    test_y_blocks: list[np.ndarray] = []
    test_base_blocks: list[np.ndarray] = []

    for uid, idx in train_df.groupby("user_id", sort=False).indices.items():
        train_state_blocks.append(states_train_flat[idx])
        train_y_blocks.append(y_train_flat[idx])
        train_base_blocks.append(train_pred_array[idx])
    for uid, idx in test_df.groupby("user_id", sort=False).indices.items():
        test_state_blocks.append(states_test_flat[idx])
        test_y_blocks.append(y_test_flat[idx])
        test_base_blocks.append(test_pred_array[idx])

    hawkes = fit_pooled_additive_multi_kernel_hawkes(
        state_blocks=train_state_blocks,
        y_blocks=train_y_blocks,
        base_blocks=train_base_blocks,
        half_lives=HAWKES_HALF_LIVES,
        feature_names=HAWKES_FEATURES,
        alpha_l2=1e-4,
        learn_base_scale=True,
        scale_l2=10.0,
        scale_init=1.0,
        max_iter=300,
    )

    test_preds: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    for states, y, base in zip(test_state_blocks, test_y_blocks, test_base_blocks):
        lam, _ = predict_pooled_additive_multi_kernel_hawkes(
            hawkes,
            states=states,
            base_lambda=base,
        )
        test_preds.append(lam)
        test_targets.append(y)

    return evaluate_count_forecast(np.concatenate(test_targets), np.concatenate(test_preds))["mean_poisson_nll"]


def evaluate_gbdt(
    full_df_subset: pd.DataFrame,
    block_start: pd.Timestamp,
    block_end: pd.Timestamp,
    train_end: pd.Timestamp,
) -> float:
    feature_table = build_feature_tables(
        full_df=full_df_subset,
        analysis_start=block_start,
        analysis_end=block_end,
        split_date=train_end,
        target_col=TARGET_COL,
        source_features=list(SOURCE_FEATURES),
    )
    model = fit_global_poisson_gbdt(feature_table, **GBDT_PARAMS)
    test_pred = np.clip(model.predict(feature_table.x_test), 1e-8, None)
    test_target = feature_table.row_index_test["target"].to_numpy(dtype=float)
    return evaluate_count_forecast(test_target, test_pred)["mean_poisson_nll"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3-week blockwise cross-validation across all models")
    parser.add_argument(
        "--data-path",
        default="data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        help="Path to daily grid csv",
    )
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/blockwise_cv",
        help="Directory for cross-validation artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data once...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES, *SOURCE_FEATURES]))
    full_df = load_daily_grid(args.data_path, value_cols=cols)
    print(f"  loaded {len(full_df):,} rows, {full_df['user_id'].nunique():,} users")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    blocks = build_blocks(GLOBAL_START, GLOBAL_END, BLOCK_LEN)
    print(f"  built {len(blocks)} blocks")

    print("Building Hawkes states cache once for all blocks...")
    hawkes_cache = build_user_states_cache(
        full_df, features=HAWKES_FEATURES, half_lives=HAWKES_HALF_LIVES,
    )

    results: list[dict] = []

    for block in blocks:
        block_idx = block["block_idx"]
        block_start = block["block_start"]
        block_end = block["block_end"]
        train_end = block["train_end"]
        test_start = block["test_start"]

        block_df = filter_date_range(full_df, start_date=block_start, end_date=block_end)
        if block_df.empty:
            continue

        train_df = block_df[block_df["event_date"] <= train_end].copy()
        test_df = block_df[block_df["event_date"] >= test_start].copy()

        train_daily_mean = train_df.groupby("event_date")[TARGET_COL].mean().sort_index()

        block_label = f"{block_start.date()}..{block_end.date()}"
        print(f"\n[Block {block_idx + 1}/{len(blocks)}] {block_label} (train {len(train_df):,} / test {len(test_df):,})")

        t0 = time.time()
        nll_global = evaluate_global_poisson(train_df, test_df)
        print(f"  Global Poisson:           NLL = {nll_global:.4f}  ({time.time() - t0:.1f}s)")

        t0 = time.time()
        nll_rolling = evaluate_rolling(daily_mean_full, test_df)
        print(f"  Rolling Poisson:          NLL = {nll_rolling:.4f}  ({time.time() - t0:.1f}s)")

        t0 = time.time()
        nll_rs, rs_model = evaluate_rolling_seasonal(train_daily_mean, daily_mean_full, test_df)
        print(f"  Rolling Seasonal:         NLL = {nll_rs:.4f}  ({time.time() - t0:.1f}s)")

        t0 = time.time()
        nll_pers, train_pers_pred, test_pers_pred, _ = evaluate_personalized(train_df, test_df, rs_model)
        print(f"  Personalized:             NLL = {nll_pers:.4f}  ({time.time() - t0:.1f}s)")

        t0 = time.time()
        try:
            nll_hawkes = evaluate_hawkes(
                full_df=full_df,
                train_df=train_df,
                test_df=test_df,
                block_start=block_start,
                block_end=block_end,
                train_end=train_end,
                train_personalized_pred=train_pers_pred,
                test_personalized_pred=test_pers_pred,
                cache=hawkes_cache,
            )
            print(f"  Scaled-baseline Hawkes:   NLL = {nll_hawkes:.4f}  ({time.time() - t0:.1f}s)")
        except Exception as exc:  # pragma: no cover - defensive
            nll_hawkes = float("nan")
            print(f"  Scaled-baseline Hawkes:   FAILED ({exc})")

        t0 = time.time()
        try:
            nll_gbdt = evaluate_gbdt(
                full_df_subset=full_df.loc[:, ["user_id", "event_date", *list(dict.fromkeys([*SOURCE_FEATURES, TARGET_COL]))]].copy(),
                block_start=block_start,
                block_end=block_end,
                train_end=train_end,
            )
            print(f"  GBDT (experimental):      NLL = {nll_gbdt:.4f}  ({time.time() - t0:.1f}s)")
        except Exception as exc:  # pragma: no cover - defensive
            nll_gbdt = float("nan")
            print(f"  GBDT (experimental):      FAILED ({exc})")

        results.append(
            {
                "block_idx": block_idx,
                "block_label": block_label,
                "block_start": str(block_start.date()),
                "block_end": str(block_end.date()),
                "test_rows": int(len(test_df)),
                "test_target_mean": float(test_df[TARGET_COL].mean()),
                "Global Poisson": nll_global,
                "Rolling Poisson": nll_rolling,
                "Rolling Seasonal": nll_rs,
                "Personalized Gamma-Poisson": nll_pers,
                "Scaled-baseline Hawkes": nll_hawkes,
                "GBDT (experimental)": nll_gbdt,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "cv_results.csv", index=False)

    model_labels = [
        "Global Poisson",
        "Rolling Poisson",
        "Rolling Seasonal",
        "Personalized Gamma-Poisson",
        "Scaled-baseline Hawkes",
        "GBDT (experimental)",
    ]

    summary = {
        "n_blocks": int(len(df)),
        "block_len_days": int(BLOCK_LEN),
        "train_len_days": int(TRAIN_LEN),
        "test_len_days": int(BLOCK_LEN - TRAIN_LEN),
        "models": {},
    }
    for label in model_labels:
        vals = df[label].dropna().to_numpy(dtype=float)
        summary["models"][label] = {
            "n": int(len(vals)),
            "mean_nll": float(np.mean(vals)),
            "median_nll": float(np.median(vals)),
            "std_nll": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min_nll": float(np.min(vals)),
            "max_nll": float(np.max(vals)),
        }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Strip plot
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    rng = np.random.default_rng(0)
    colors = ["#2E5EAA"] * 5 + ["#D2691E"]
    for i, label in enumerate(model_labels):
        vals = df[label].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        x_jitter = i + (rng.random(len(vals)) - 0.5) * 0.32
        ax.scatter(x_jitter, vals, color=colors[i], alpha=0.65, s=42, edgecolors="white", linewidths=0.6)
        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        ax.hlines(mean_val, i - 0.28, i + 0.28, color="#0B3C5D", linewidth=2.4, label="Mean" if i == 0 else None, zorder=4)
        ax.hlines(median_val, i - 0.28, i + 0.28, color="#D2691E", linewidth=1.8, linestyles="--", label="Median" if i == 0 else None, zorder=4)
        ax.text(i, mean_val, f"{mean_val:.4f}", ha="center", va="bottom", fontsize=9, color="#0B3C5D", fontweight="bold")

    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels([lbl.replace(" ", "\n", 1) for lbl in model_labels], fontsize=9)
    ax.set_ylabel("Test NLL per user-day (lower is better)")
    ax.set_title(f"Blockwise 3-week CV: per-block NLL across {len(df)} blocks (14d train / 7d test)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "cv_strip_plot.png", dpi=150)
    plt.close(fig)

    print("\nDone. Mean NLL per model:")
    for label in model_labels:
        m = summary["models"][label]
        print(f"  {label:<32s} mean={m['mean_nll']:.4f}  median={m['median_nll']:.4f}  std={m['std_nll']:.4f}  (n={m['n']})")


if __name__ == "__main__":
    main()

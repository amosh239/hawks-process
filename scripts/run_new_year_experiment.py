"""Predict the previously-excluded new-year period (Jan 1-14) using models
trained on the stable Jan 15 - Aug 9 period.

Goal: see which models in the ladder can adapt to a different regime (early
January with elevated purchasing) and which cannot.

The test period precedes the train period in calendar time. Models that rely
on a constant fitted level (Global Poisson, Seasonal Poisson) cannot adapt.
Models that use recent history (Rolling Poisson, Rolling Seasonal, Personalized,
Hawkes, GBDT) start with no prior history at Jan 1 and accumulate it as the
test progresses.

Outputs:
  diploma/reports/new_year_experiment/per_day_predictions.csv
  diploma/reports/new_year_experiment/per_model_test_nll.csv
  diploma/reports/new_year_experiment/per_day_predictions.png
  diploma/reports/new_year_experiment/per_model_nll_bar.png
  diploma/reports/new_year_experiment/summary.json
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

from src.diploma_baselines.data import filter_date_range, load_daily_grid
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalPoissonModel,
    GlobalRollingMeanPoissonModel,
    GlobalRollingSeasonalPoissonModel,
    GlobalSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_basis_states,
    fit_pooled_additive_multi_kernel_hawkes,
    predict_pooled_additive_multi_kernel_hawkes,
)
from src.diploma_experimental.gbdt import (
    SOURCE_FEATURES,
    build_feature_tables,
    fit_global_poisson_gbdt,
)


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

TRAIN_START = pd.Timestamp("2025-01-15")
TRAIN_END = pd.Timestamp("2025-08-09")
NEW_YEAR_START = pd.Timestamp("2025-01-01")
NEW_YEAR_END = pd.Timestamp("2025-01-14")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict new-year (Jan 1-14) using models trained on Jan 15 - Aug 9")
    parser.add_argument(
        "--data-path",
        default="data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        help="Path to daily grid csv",
    )
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/new_year_experiment",
        help="Directory for output artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES, *SOURCE_FEATURES]))
    full_df = load_daily_grid(args.data_path, value_cols=cols)
    print(f"  loaded {len(full_df):,} rows")

    train_df = filter_date_range(full_df, start_date=TRAIN_START, end_date=TRAIN_END)
    test_df = filter_date_range(full_df, start_date=NEW_YEAR_START, end_date=NEW_YEAR_END)
    print(f"  train rows = {len(train_df):,}  ({TRAIN_START.date()} → {TRAIN_END.date()})")
    print(f"  test rows  = {len(test_df):,}  ({NEW_YEAR_START.date()} → {NEW_YEAR_END.date()})")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    train_daily_mean = train_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    test_y = test_df[TARGET_COL].to_numpy(dtype=float)

    predictions: dict[str, np.ndarray] = {}

    # === 1. Global Poisson ===
    print("\nFitting Global Poisson...")
    gp = GlobalPoissonModel().fit(train_df[TARGET_COL].to_numpy())
    predictions["Global Poisson"] = gp.predict(len(test_df))

    # === 2. Seasonal Poisson ===
    print("Fitting Seasonal Poisson...")
    sp = GlobalSeasonalPoissonModel().fit(
        train_df[TARGET_COL].to_numpy(),
        train_df["dow"].to_numpy(),
    )
    predictions["Seasonal Poisson"] = sp.predict(test_df["dow"].to_numpy())

    train_mean = float(train_df[TARGET_COL].mean())

    # === 3. Rolling Poisson ===
    print("Fitting Rolling Poisson...")
    rp = GlobalRollingMeanPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(daily_mean_full)
    # Bypass internal validation: read daily_prediction_ directly, fill NaN with train mean
    rp_pred_series = rp.daily_prediction_.reindex(pd.to_datetime(test_df["event_date"].unique())).fillna(train_mean)
    rp_lookup = {pd.Timestamp(d).normalize(): float(rp_pred_series.loc[pd.Timestamp(d).normalize()]) for d in rp_pred_series.index}
    predictions["Rolling Poisson"] = np.array(
        [rp_lookup[pd.Timestamp(d).normalize()] for d in test_df["event_date"].to_numpy()],
        dtype=float,
    )

    # === 4. Rolling Seasonal Poisson ===
    print("Fitting Rolling Seasonal Poisson...")
    rsp = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean,
        daily_mean_full,
    )
    # Bypass internal validation: combine level (with NaN→train_mean fallback) with seasonal profile
    test_unique_dates = pd.DatetimeIndex(pd.to_datetime(test_df["event_date"].unique()))
    level_series = rsp.level_prediction_.reindex(test_unique_dates).fillna(train_mean)
    test_dows = test_unique_dates.dayofweek.astype(int).to_numpy()
    rsp_pred_series = pd.Series(
        level_series.to_numpy(dtype=float) * rsp.seasonal_profile_[test_dows],
        index=test_unique_dates,
    )
    rsp_lookup = {pd.Timestamp(d).normalize(): float(rsp_pred_series.loc[pd.Timestamp(d).normalize()]) for d in rsp_pred_series.index}
    predictions["Rolling Seasonal"] = np.array(
        [rsp_lookup[pd.Timestamp(d).normalize()] for d in test_df["event_date"].to_numpy()],
        dtype=float,
    )

    # === 5. Personalized Gamma-Poisson ===
    print("Fitting Personalized Gamma-Poisson...")
    base_train = rsp.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    base_test = np.array(
        [rsp_lookup[pd.Timestamp(d).normalize()] for d in test_df["event_date"].to_numpy()],
        dtype=float,
    )
    scaler = PersonalizedGammaPoissonScaler().fit(
        train_df["user_id"].to_numpy(),
        train_df[TARGET_COL].to_numpy(),
        base_train,
    )
    predictions["Personalized Gamma-Poisson"] = scaler.predict(
        test_df["user_id"].to_numpy(), base_test, method="posterior_mean"
    )

    # === 6. Scaled-baseline Hawkes ===
    print("Fitting Scaled-baseline Hawkes...")
    beta = np.log(2.0) / np.asarray(HALF_LIVES, dtype=float)
    train_pers = scaler.predict(train_df["user_id"].to_numpy(), base_train, method="posterior_mean")
    test_pers = predictions["Personalized Gamma-Poisson"]

    train_pers_by_user: dict[int, np.ndarray] = {}
    for uid, idx in train_df.groupby("user_id", sort=False).indices.items():
        train_pers_by_user[int(uid)] = train_pers[idx]
    test_pers_by_user: dict[int, np.ndarray] = {}
    for uid, idx in test_df.groupby("user_id", sort=False).indices.items():
        test_pers_by_user[int(uid)] = test_pers[idx]

    train_state_blocks: list[np.ndarray] = []
    train_y_blocks: list[np.ndarray] = []
    train_base_blocks: list[np.ndarray] = []
    test_state_blocks_by_user: dict[int, np.ndarray] = {}

    train_start64 = np.datetime64(TRAIN_START)
    train_end64 = np.datetime64(TRAIN_END)
    new_year_start64 = np.datetime64(NEW_YEAR_START)
    new_year_end64 = np.datetime64(NEW_YEAR_END)

    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        full_dates = full_user["event_date"].to_numpy(dtype="datetime64[ns]")
        full_y = full_user[TARGET_COL].to_numpy(dtype=float)

        train_mask = (full_dates >= train_start64) & (full_dates <= train_end64)
        test_mask = (full_dates >= new_year_start64) & (full_dates <= new_year_end64)

        if train_mask.any():
            base_t = train_pers_by_user.get(int(user_id))
            if base_t is not None:
                train_state_blocks.append(states_full[train_mask])
                train_y_blocks.append(full_y[train_mask])
                train_base_blocks.append(base_t)

        if test_mask.any():
            test_state_blocks_by_user[int(user_id)] = states_full[test_mask]

    hawkes = fit_pooled_additive_multi_kernel_hawkes(
        state_blocks=train_state_blocks,
        y_blocks=train_y_blocks,
        base_blocks=train_base_blocks,
        half_lives=HALF_LIVES,
        feature_names=HAWKES_FEATURES,
        alpha_l2=1e-4,
        learn_base_scale=True,
        scale_l2=10.0,
        scale_init=1.0,
        max_iter=300,
    )
    print(f"  Hawkes: c={hawkes.base_scale:.4f}, ||alpha||_2={float(np.linalg.norm(hawkes.alpha)):.4f}")

    hawkes_pred = np.zeros(len(test_df), dtype=float)
    for uid, idx in test_df.groupby("user_id", sort=False).indices.items():
        states = test_state_blocks_by_user.get(int(uid))
        if states is None:
            hawkes_pred[idx] = test_pers[idx]
            continue
        base = test_pers[idx]
        lam, _ = predict_pooled_additive_multi_kernel_hawkes(
            hawkes,
            states=states,
            base_lambda=base,
        )
        hawkes_pred[idx] = lam
    predictions["Scaled-baseline Hawkes"] = hawkes_pred

    # === 7. GBDT ===
    print("Fitting GBDT (custom split for new-year reverse test)...")
    # Trick: build features over Jan 1 - Aug 9 with split_date=Jan 14;
    # build_feature_tables puts dates <= Jan 14 in train, > Jan 14 in test.
    # We then SWAP: use the function's "test" rows (= our train) to fit and
    # its "train" rows (= our test) to evaluate.
    feature_table = build_feature_tables(
        full_df=full_df.loc[
            :,
            ["user_id", "event_date", *list(dict.fromkeys([*SOURCE_FEATURES, TARGET_COL]))],
        ].copy(),
        analysis_start=NEW_YEAR_START,
        analysis_end=TRAIN_END,
        split_date=NEW_YEAR_END,
        target_col=TARGET_COL,
        source_features=list(SOURCE_FEATURES),
    )
    # The function's "train" = Jan 1-14 (our test), "test" = Jan 15 - Aug 9 (our train).
    gbdt_train_x = feature_table.x_test
    gbdt_train_y = feature_table.y_test
    gbdt_test_x = feature_table.x_train
    gbdt_test_y = feature_table.y_train
    gbdt_test_index = feature_table.row_index_train

    # Sanity check: gbdt_test_y should match test_df target by user_id, event_date.
    print(
        f"  GBDT train rows = {len(gbdt_train_x):,}  test rows = {len(gbdt_test_x):,}"
        f"  test mean = {gbdt_test_y.mean():.4f}"
    )

    gbdt_model = fit_global_poisson_gbdt(
        type("FT", (), {"x_train": gbdt_train_x, "y_train": gbdt_train_y})(),
        seed=42,
        max_depth=5,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=40,
    )
    gbdt_pred_raw = np.clip(gbdt_model.predict(gbdt_test_x), 1e-8, None)

    # Need to align gbdt predictions with test_df row order
    pred_index_df = gbdt_test_index.copy()
    pred_index_df["gbdt_pred"] = gbdt_pred_raw
    aligned = test_df.merge(pred_index_df[["user_id", "event_date", "gbdt_pred"]], on=["user_id", "event_date"], how="left")
    predictions["GBDT (experimental)"] = aligned["gbdt_pred"].fillna(float(train_df[TARGET_COL].mean())).to_numpy(dtype=float)

    # === Compute test NLL per model ===
    metrics_per_model: dict[str, dict[str, float]] = {}
    for label, pred in predictions.items():
        m = evaluate_count_forecast(test_y, pred)
        metrics_per_model[label] = {
            "test_mean_poisson_nll": float(m["mean_poisson_nll"]),
            "test_poisson_loglik": float(m["poisson_loglik"]),
            "test_mae": float(m["mae"]),
            "test_rmse": float(m["rmse"]),
            "test_aggregate_bias": float(m["aggregate_bias"]),
            "test_relative_aggregate_bias": float(m["relative_aggregate_bias"]),
            "test_mean_prediction": float(m["mean_prediction"]),
        }

    # === Per-day predictions for plotting ===
    test_with_pred = test_df.loc[:, ["user_id", "event_date", TARGET_COL]].copy()
    for label, pred in predictions.items():
        test_with_pred[label] = pred
    daily_means = test_with_pred.groupby("event_date").mean(numeric_only=True).reset_index()
    daily_means = daily_means.rename(columns={TARGET_COL: "actual_mean_target"})
    daily_means.to_csv(output_dir / "per_day_predictions.csv", index=False)

    # Per-day NLL
    per_day_nll: dict[pd.Timestamp, dict[str, float]] = {}
    for date, group in test_with_pred.groupby("event_date"):
        y_d = group[TARGET_COL].to_numpy(dtype=float)
        per_day_nll[pd.Timestamp(date)] = {}
        for label, _ in predictions.items():
            p_d = group[label].to_numpy(dtype=float)
            per_day_nll[pd.Timestamp(date)][label] = float(evaluate_count_forecast(y_d, p_d)["mean_poisson_nll"])
    per_day_nll_df = pd.DataFrame(per_day_nll).T.reset_index().rename(columns={"index": "event_date"})
    per_day_nll_df.to_csv(output_dir / "per_day_nll.csv", index=False)

    # Save per-model summary
    summary_df = pd.DataFrame(metrics_per_model).T.reset_index().rename(columns={"index": "model"})
    summary_df.to_csv(output_dir / "per_model_test_nll.csv", index=False)

    # Save full summary JSON
    summary = {
        "train_window": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "test_window": [str(NEW_YEAR_START.date()), str(NEW_YEAR_END.date())],
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_mean_target": float(test_y.mean()),
        "train_mean_target": float(train_df[TARGET_COL].mean()),
        "models": metrics_per_model,
        "hawkes_params": {
            "c": float(hawkes.base_scale),
            "alpha_norm": float(np.linalg.norm(hawkes.alpha)),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # === Plot 1. Per-day mean prediction overlaid on actual ===
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    daily_means_sorted = daily_means.sort_values("event_date")
    ax.plot(
        daily_means_sorted["event_date"],
        daily_means_sorted["actual_mean_target"],
        marker="o",
        color="#222222",
        linewidth=2.0,
        label="Actual mean target",
        zorder=4,
    )
    palette = {
        "Global Poisson": "#888888",
        "Seasonal Poisson": "#A9A9A9",
        "Rolling Poisson": "#0B3C5D",
        "Rolling Seasonal": "#2E5EAA",
        "Personalized Gamma-Poisson": "#2E8B57",
        "Scaled-baseline Hawkes": "#D2691E",
        "GBDT (experimental)": "#B56576",
    }
    for label in predictions.keys():
        ax.plot(
            daily_means_sorted["event_date"],
            daily_means_sorted[label],
            marker=".",
            linewidth=1.4,
            color=palette.get(label, "#444444"),
            label=label,
        )
    ax.axhline(
        float(train_df[TARGET_COL].mean()),
        color="#888888",
        linestyle=":",
        linewidth=1.0,
        label=f"Train mean = {float(train_df[TARGET_COL].mean()):.4f}",
    )
    ax.set_xlabel("Date (new-year period)")
    ax.set_ylabel("Mean prediction per user-day")
    ax.set_title("Per-day mean prediction during new-year (Jan 1-14)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "per_day_predictions.png", dpi=150)
    plt.close(fig)

    # === Plot 2. Aggregate test NLL per model (bar chart) ===
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ordered_labels = list(predictions.keys())
    nlls = [metrics_per_model[lbl]["test_mean_poisson_nll"] for lbl in ordered_labels]
    colors = [palette.get(lbl, "#444444") for lbl in ordered_labels]
    bars = ax.bar(range(len(ordered_labels)), nlls, color=colors, edgecolor="white")
    for rect, val in zip(bars, nlls):
        ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0B3C5D")
    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels([lbl.replace(" ", "\n", 1) for lbl in ordered_labels], fontsize=9)
    ax.set_ylabel("Test NLL per user-day on Jan 1-14 (lower is better)")
    ax.set_title("New-year test NLL by model (lower is better)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "per_model_nll_bar.png", dpi=150)
    plt.close(fig)

    # === Plot 3. Per-day NLL evolution ===
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    nll_df_sorted = per_day_nll_df.copy()
    nll_df_sorted["event_date"] = pd.to_datetime(nll_df_sorted["event_date"])
    nll_df_sorted = nll_df_sorted.sort_values("event_date")
    for label in predictions.keys():
        ax.plot(
            nll_df_sorted["event_date"],
            nll_df_sorted[label],
            marker=".",
            linewidth=1.6,
            color=palette.get(label, "#444444"),
            label=label,
        )
    ax.set_xlabel("Date (new-year period)")
    ax.set_ylabel("Per-day test NLL (lower is better)")
    ax.set_title("Per-day test NLL evolution: who adapts faster")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "per_day_nll.png", dpi=150)
    plt.close(fig)

    print("\nPer-model new-year test NLL:")
    for lbl in ordered_labels:
        m = metrics_per_model[lbl]
        print(
            f"  {lbl:<32s}"
            f"  NLL={m['test_mean_poisson_nll']:.4f}"
            f"  bias={m['test_aggregate_bias']:+.4f}"
            f"  rel_bias={m['test_relative_aggregate_bias']*100:+.1f}%"
        )


if __name__ == "__main__":
    main()

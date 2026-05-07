"""Sweep alpha_l2 regularization for the chapter-6 scaled-baseline Hawkes.

Runs the same protocol as scripts/run_experimental_1_hawkes.py (main train / test split,
half-lives 1 and 3 days, learn_base_scale=True, scale_l2=10) but iterates the
alpha_l2 hyperparameter over a small grid. For each value collects:
  - test NLL per user-day,
  - learned base_scale c,
  - the full alpha matrix (5 features x 2 half-lives).

Outputs:
  diploma/reports/hawkes_alpha_l2_sweep/sweep_results.csv
  diploma/reports/hawkes_alpha_l2_sweep/alpha_matrices.csv
  diploma/reports/hawkes_alpha_l2_sweep/test_nll_vs_alpha_l2.png
  diploma/reports/hawkes_alpha_l2_sweep/alpha_heatmaps.png
  diploma/reports/hawkes_alpha_l2_sweep/summary.json
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


HALF_LIVES = (1.0, 3.0)
FEATURES = tuple(FEATURE_NAMES)
TARGET_COL = "to_ord"
ANALYSIS_START = pd.Timestamp("2025-01-15")
ANALYSIS_END = pd.Timestamp("2025-09-30")
TRAIN_RATIO = 0.8
WINDOW_SIZE = 7
SCALE_L2 = 10.0
SCALE_INIT = 1.0
MAX_ITER = 300

ALPHA_L2_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep alpha_l2 for the chapter-6 Hawkes model")
    parser.add_argument(
        "--data-path",
        default="data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        help="Path to daily grid csv",
    )
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/hawkes_alpha_l2_sweep",
        help="Directory for sweep artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *FEATURES]))
    full_df = load_daily_grid(args.data_path, value_cols=cols)
    print(f"  loaded {len(full_df):,} rows")

    analysis_df = filter_date_range(full_df, start_date=ANALYSIS_START, end_date=ANALYSIS_END)
    split = split_panel_by_date(analysis_df, train_ratio=TRAIN_RATIO)
    train_df = split.train
    test_df = split.test

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()
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
    train_pers = scaler.predict(train_df["user_id"].to_numpy(), base_train, method="posterior_mean")
    test_pers = scaler.predict(test_df["user_id"].to_numpy(), base_test, method="posterior_mean")

    baseline_test_metrics = evaluate_count_forecast(test_df[TARGET_COL].to_numpy(), test_pers)
    baseline_test_nll = float(baseline_test_metrics["mean_poisson_nll"])
    print(f"  personalized baseline test NLL = {baseline_test_nll:.5f}")

    beta = np.log(2.0) / np.asarray(HALF_LIVES, dtype=float)

    print("Building Hawkes states once...")
    full_groups = full_df.groupby("user_id", sort=False)
    train_user_groups = train_df.groupby("user_id", sort=False).indices
    test_user_groups = test_df.groupby("user_id", sort=False).indices

    train_state_blocks: list[np.ndarray] = []
    train_y_blocks: list[np.ndarray] = []
    train_base_blocks: list[np.ndarray] = []
    test_state_blocks: list[np.ndarray] = []
    test_y_blocks: list[np.ndarray] = []
    test_base_blocks: list[np.ndarray] = []

    train_pred_by_user: dict[int, np.ndarray] = {}
    test_pred_by_user: dict[int, np.ndarray] = {}
    for uid, idx in train_user_groups.items():
        train_pred_by_user[int(uid)] = train_pers[idx]
    for uid, idx in test_user_groups.items():
        test_pred_by_user[int(uid)] = test_pers[idx]

    train_start64 = np.datetime64(train_df["event_date"].min())
    train_end64 = np.datetime64(train_df["event_date"].max())
    test_start64 = np.datetime64(test_df["event_date"].min())
    test_end64 = np.datetime64(test_df["event_date"].max())

    for user_id, full_user in full_groups:
        x_full = full_user.loc[:, list(FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        full_dates = full_user["event_date"].to_numpy(dtype="datetime64[ns]")
        train_mask = (full_dates >= train_start64) & (full_dates <= train_end64)
        test_mask = (full_dates >= test_start64) & (full_dates <= test_end64)

        if train_mask.any():
            base_t = train_pred_by_user.get(int(user_id))
            if base_t is None:
                continue
            train_state_blocks.append(states_full[train_mask])
            train_y_blocks.append(full_user[TARGET_COL].to_numpy(dtype=float)[train_mask])
            train_base_blocks.append(base_t)

        if test_mask.any():
            base_t = test_pred_by_user.get(int(user_id))
            if base_t is None:
                continue
            test_state_blocks.append(states_full[test_mask])
            test_y_blocks.append(full_user[TARGET_COL].to_numpy(dtype=float)[test_mask])
            test_base_blocks.append(base_t)

    test_y_concat = np.concatenate(test_y_blocks)

    sweep_rows: list[dict] = []
    alpha_matrices: dict[float, np.ndarray] = {}

    for alpha_l2 in ALPHA_L2_GRID:
        t0 = time.time()
        hawkes = fit_pooled_additive_multi_kernel_hawkes(
            state_blocks=train_state_blocks,
            y_blocks=train_y_blocks,
            base_blocks=train_base_blocks,
            half_lives=HALF_LIVES,
            feature_names=FEATURES,
            alpha_l2=float(alpha_l2),
            learn_base_scale=True,
            scale_l2=SCALE_L2,
            scale_init=SCALE_INIT,
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
        test_pred = np.concatenate(test_preds)
        metrics = evaluate_count_forecast(test_y_concat, test_pred)

        alpha_matrix = hawkes.alpha_matrix()
        alpha_matrices[float(alpha_l2)] = alpha_matrix
        alpha_norm = float(np.linalg.norm(alpha_matrix))

        sweep_rows.append(
            {
                "alpha_l2": float(alpha_l2),
                "test_poisson_loglik": float(metrics["poisson_loglik"]),
                "test_mean_poisson_nll": float(metrics["mean_poisson_nll"]),
                "test_mae": float(metrics["mae"]),
                "test_rmse": float(metrics["rmse"]),
                "test_relative_aggregate_bias": float(metrics["relative_aggregate_bias"]),
                "learned_base_scale": float(hawkes.base_scale),
                "alpha_l2_norm": alpha_norm,
                "fit_success": bool(hawkes.success),
                "fit_seconds": float(time.time() - t0),
            }
        )
        print(
            f"  alpha_l2={alpha_l2:>7.0e}  test NLL={metrics['mean_poisson_nll']:.5f}  "
            f"c={hawkes.base_scale:.4f}  ||alpha||_2={alpha_norm:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )

    df = pd.DataFrame(sweep_rows)
    df.to_csv(output_dir / "sweep_results.csv", index=False)

    # Long-form alpha table
    long_rows = []
    for alpha_l2, m in alpha_matrices.items():
        for fi, feature in enumerate(FEATURES):
            for hi, hl in enumerate(HALF_LIVES):
                long_rows.append(
                    {
                        "alpha_l2": float(alpha_l2),
                        "feature": feature,
                        "half_life": float(hl),
                        "alpha": float(m[fi, hi]),
                    }
                )
    pd.DataFrame(long_rows).to_csv(output_dir / "alpha_matrices.csv", index=False)

    # Plot 1: test NLL vs alpha_l2
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    xs = [max(row["alpha_l2"], 1e-7) for row in sweep_rows]  # 0 -> 1e-7 for log scale display
    nll_values = [row["test_mean_poisson_nll"] for row in sweep_rows]
    ax.semilogx(xs, nll_values, marker="o", color="#0B3C5D", linewidth=1.8, label="Hawkes")
    ax.axhline(
        baseline_test_nll,
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        label=f"Personalized baseline NLL = {baseline_test_nll:.5f}",
    )
    for x, val, raw in zip(xs, nll_values, sweep_rows):
        ax.text(x, val, f"{val:.5f}", ha="center", va="bottom", fontsize=9, color="#0B3C5D")
    xticks_real = [row["alpha_l2"] for row in sweep_rows]
    xticks_display = [max(v, 1e-7) for v in xticks_real]
    ax.set_xticks(xticks_display)
    ax.set_xticklabels([("0" if v == 0 else f"{v:.0e}") for v in xticks_real])
    ax.set_xlabel("alpha_l2 (log scale)")
    ax.set_ylabel("Test NLL per user-day (lower is better)")
    ax.set_title("Чувствительность Scaled-baseline Hawkes к alpha_l2")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "test_nll_vs_alpha_l2.png", dpi=150)
    plt.close(fig)

    # Plot 2: alpha heatmaps for representative subset of alpha_l2
    heatmap_grid = (0.0, 1e-4, 1.0, 1e3, 1e5, 1e6)
    heatmap_grid = tuple(v for v in heatmap_grid if float(v) in alpha_matrices)
    n_panels = len(heatmap_grid)
    vmax = max(float(np.max(alpha_matrices[float(v)])) for v in heatmap_grid)
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 4.2), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]
    for ax, alpha_l2 in zip(axes, heatmap_grid):
        m = alpha_matrices[float(alpha_l2)]
        im = ax.imshow(m, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(HALF_LIVES)))
        ax.set_xticklabels([str(int(h)) for h in HALF_LIVES])
        ax.set_yticks(range(len(FEATURES)))
        ax.set_yticklabels(FEATURES)
        ax.set_title(f"α_l2 = {'0' if alpha_l2 == 0 else f'{alpha_l2:.0e}'}", fontsize=10)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#111111")
        ax.set_xlabel("half-life")
        if ax is axes[0]:
            ax.set_ylabel("feature")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="alpha")
    fig.suptitle("Hawkes alpha-matrix на представительных значениях alpha_l2", fontsize=12)
    fig.savefig(output_dir / "alpha_heatmaps.png", dpi=150)
    plt.close(fig)

    # Plot 3: ||alpha||_2 and base_scale c vs alpha_l2
    fig, ax1 = plt.subplots(figsize=(9.0, 4.6))
    norm_values = [row["alpha_l2_norm"] for row in sweep_rows]
    scale_values = [row["learned_base_scale"] for row in sweep_rows]
    ax1.semilogx(xs, norm_values, marker="o", color="#0B3C5D", linewidth=1.8, label="||α||_2")
    ax1.set_xlabel("alpha_l2 (log scale)")
    ax1.set_ylabel("||α||_2", color="#0B3C5D")
    ax1.tick_params(axis="y", labelcolor="#0B3C5D")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.semilogx(xs, scale_values, marker="s", color="#D2691E", linewidth=1.4, label="learned base scale c")
    ax2.set_ylabel("learned base scale c", color="#D2691E")
    ax2.tick_params(axis="y", labelcolor="#D2691E")

    ax1.set_xticks(xticks_display)
    ax1.set_xticklabels([("0" if v == 0 else f"{v:.0e}") for v in xticks_real])
    ax1.set_title("Структура решения: ||α||_2 и base_scale c от alpha_l2")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_norm_and_scale_vs_alpha_l2.png", dpi=150)
    plt.close(fig)

    summary = {
        "personalized_baseline_test_nll": baseline_test_nll,
        "alpha_l2_grid": [float(v) for v in ALPHA_L2_GRID],
        "results": sweep_rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"  {output_dir}/sweep_results.csv")
    print(f"  {output_dir}/alpha_matrices.csv")
    print(f"  {output_dir}/test_nll_vs_alpha_l2.png")
    print(f"  {output_dir}/alpha_heatmaps.png")
    print(f"  {output_dir}/summary.json")


if __name__ == "__main__":
    main()

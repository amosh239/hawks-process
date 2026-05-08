"""Two regularization sensitivity sweeps for the chapter-6 scaled-baseline Hawkes.

Sweep A: alpha_l2 ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1}, scale_l2 = 0
Sweep B: scale_l2 ∈ {0, 1e-1, 1, 10, 100},                    alpha_l2 = 0

Each sweep produces one plot showing test NLL across the grid, plus
side-by-side learned (c, ||alpha||) values.

Output:
  diploma/reports/hawkes_reg_sweeps/sweep_alpha_l2.csv
  diploma/reports/hawkes_reg_sweeps/sweep_scale_l2.csv
  diploma/reports/hawkes_reg_sweeps/alpha_l2_grid.png
  diploma/reports/hawkes_reg_sweeps/scale_l2_grid.png
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
MAX_ITER = 300

ALPHA_L2_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SCALE_L2_GRID = (0.0, 1e-1, 1.0, 10.0, 100.0)

OUT_DIR = Path("diploma/reports/hawkes_reg_sweeps")


def prepare_inputs():
    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    analysis_df = filter_date_range(full_df, start_date=ANALYSIS_START, end_date=ANALYSIS_END)
    split = split_panel_by_date(analysis_df, train_ratio=TRAIN_RATIO)
    train_df = split.train
    test_df = split.test

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    train_daily_mean = train_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean, daily_mean_full
    )
    base_train = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    base_test = rs.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)

    scaler = PersonalizedGammaPoissonScaler().fit(
        train_df["user_id"].to_numpy(),
        train_df[TARGET_COL].to_numpy(),
        base_train,
    )
    train_pers = scaler.predict(train_df["user_id"].to_numpy(), base_train, method="posterior_mean")
    test_pers = scaler.predict(test_df["user_id"].to_numpy(), base_test, method="posterior_mean")

    baseline_test_nll = float(
        evaluate_count_forecast(test_df[TARGET_COL].to_numpy(), test_pers)["mean_poisson_nll"]
    )
    print(f"  personalized baseline test NLL = {baseline_test_nll:.5f}")

    beta = np.log(2.0) / np.asarray(HALF_LIVES, dtype=float)

    print("Building Hawkes states once...")
    train_user_groups = train_df.groupby("user_id", sort=False).indices
    test_user_groups = test_df.groupby("user_id", sort=False).indices

    train_pred_by_user = {int(uid): train_pers[idx] for uid, idx in train_user_groups.items()}
    test_pred_by_user = {int(uid): test_pers[idx] for uid, idx in test_user_groups.items()}

    train_start64 = np.datetime64(train_df["event_date"].min())
    train_end64 = np.datetime64(train_df["event_date"].max())
    test_start64 = np.datetime64(test_df["event_date"].min())
    test_end64 = np.datetime64(test_df["event_date"].max())

    train_state_blocks = []; train_y_blocks = []; train_base_blocks = []
    test_state_blocks = []; test_y_blocks = []; test_base_blocks = []

    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        full_dates = full_user["event_date"].to_numpy(dtype="datetime64[ns]")

        train_mask = (full_dates >= train_start64) & (full_dates <= train_end64)
        test_mask = (full_dates >= test_start64) & (full_dates <= test_end64)

        if train_mask.any():
            base_t = train_pred_by_user.get(int(user_id))
            if base_t is not None:
                train_state_blocks.append(states_full[train_mask])
                train_y_blocks.append(full_user[TARGET_COL].to_numpy(dtype=float)[train_mask])
                train_base_blocks.append(base_t)
        if test_mask.any():
            base_t = test_pred_by_user.get(int(user_id))
            if base_t is not None:
                test_state_blocks.append(states_full[test_mask])
                test_y_blocks.append(full_user[TARGET_COL].to_numpy(dtype=float)[test_mask])
                test_base_blocks.append(base_t)

    return {
        "train_state_blocks": train_state_blocks,
        "train_y_blocks": train_y_blocks,
        "train_base_blocks": train_base_blocks,
        "test_state_blocks": test_state_blocks,
        "test_y_blocks": test_y_blocks,
        "test_base_blocks": test_base_blocks,
        "test_y_concat": np.concatenate(test_y_blocks),
        "baseline_test_nll": baseline_test_nll,
    }


def run_sweep(inputs, alpha_l2_values, scale_l2_values, label):
    rows = []
    for alpha_l2, scale_l2 in zip(alpha_l2_values, scale_l2_values):
        t0 = time.time()
        hawkes = fit_pooled_additive_multi_kernel_hawkes(
            state_blocks=inputs["train_state_blocks"],
            y_blocks=inputs["train_y_blocks"],
            base_blocks=inputs["train_base_blocks"],
            half_lives=HALF_LIVES,
            feature_names=FEATURES,
            alpha_l2=float(alpha_l2),
            learn_base_scale=True,
            scale_l2=float(scale_l2),
            scale_init=1.0,
            max_iter=MAX_ITER,
        )
        test_preds = []
        for states, base in zip(inputs["test_state_blocks"], inputs["test_base_blocks"]):
            lam, _ = predict_pooled_additive_multi_kernel_hawkes(hawkes, states=states, base_lambda=base)
            test_preds.append(lam)
        metrics = evaluate_count_forecast(inputs["test_y_concat"], np.concatenate(test_preds))
        alpha_norm = float(np.linalg.norm(hawkes.alpha))
        rows.append({
            "alpha_l2": float(alpha_l2),
            "scale_l2": float(scale_l2),
            "test_mean_poisson_nll": float(metrics["mean_poisson_nll"]),
            "learned_c": float(hawkes.base_scale),
            "alpha_norm": alpha_norm,
            "fit_seconds": float(time.time() - t0),
        })
        print(
            f"  [{label}] alpha_l2={alpha_l2:>7.0e}  scale_l2={scale_l2:>5.0e}  "
            f"NLL={metrics['mean_poisson_nll']:.5f}  c={hawkes.base_scale:.4f}  ||α||={alpha_norm:.4f}"
        )
    return rows


def plot_grid(rows, sweep_var, fixed_var_name, fixed_var_value, baseline_nll, out_path, title):
    xs_raw = [r[sweep_var] for r in rows]
    xs = [max(v, xs_raw[1] / 10.0) if v == 0 else v for v in xs_raw]  # 0 → just below first nonzero, for log axis
    nlls = [r["test_mean_poisson_nll"] for r in rows]
    cs = [r["learned_c"] for r in rows]
    norms = [r["alpha_norm"] for r in rows]

    fig, (ax_nll, ax_struct) = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax_nll.semilogx(xs, nlls, marker="o", color="#0B3C5D", linewidth=1.8, label="Hawkes")
    ax_nll.axhline(baseline_nll, color="#888888", linestyle="--", linewidth=1.2,
                   label=f"Personalized baseline = {baseline_nll:.5f}")
    for x, val in zip(xs, nlls):
        ax_nll.text(x, val, f"{val:.5f}", ha="center", va="bottom", fontsize=8, color="#0B3C5D")
    ax_nll.set_xticks(xs)
    ax_nll.set_xticklabels([("0" if v == 0 else f"{v:.0e}") for v in xs_raw], fontsize=8)
    ax_nll.set_xlabel(f"{sweep_var} (log scale)")
    ax_nll.set_ylabel("Test NLL per user-day")
    ax_nll.set_title(f"Test NLL vs {sweep_var}\n({fixed_var_name} = {fixed_var_value:g})")
    ax_nll.grid(axis="y", linestyle=":", alpha=0.5)
    ax_nll.spines["top"].set_visible(False); ax_nll.spines["right"].set_visible(False)
    ax_nll.legend(frameon=False, loc="best", fontsize=9)

    ax_struct.semilogx(xs, norms, marker="o", color="#0B3C5D", linewidth=1.8, label="‖α‖₂")
    ax_struct.set_xlabel(f"{sweep_var} (log scale)")
    ax_struct.set_ylabel("‖α‖₂", color="#0B3C5D")
    ax_struct.tick_params(axis="y", labelcolor="#0B3C5D")
    ax_struct.set_xticks(xs)
    ax_struct.set_xticklabels([("0" if v == 0 else f"{v:.0e}") for v in xs_raw], fontsize=8)
    ax_struct.grid(axis="y", linestyle=":", alpha=0.5)
    ax_struct.spines["top"].set_visible(False)

    ax2 = ax_struct.twinx()
    ax2.semilogx(xs, cs, marker="s", color="#D2691E", linewidth=1.4, label="learned c")
    ax2.set_ylabel("learned c", color="#D2691E")
    ax2.tick_params(axis="y", labelcolor="#D2691E")
    ax2.spines["top"].set_visible(False)
    ax_struct.set_title(f"Структура решения: ‖α‖ и c")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = prepare_inputs()

    # Sweep A: alpha_l2 grid, scale_l2 fixed at 0
    print("\n=== Sweep A: alpha_l2 (scale_l2 = 0) ===")
    rows_a = run_sweep(
        inputs,
        alpha_l2_values=ALPHA_L2_GRID,
        scale_l2_values=[0.0] * len(ALPHA_L2_GRID),
        label="A",
    )
    pd.DataFrame(rows_a).to_csv(OUT_DIR / "sweep_alpha_l2.csv", index=False)
    plot_grid(
        rows_a, sweep_var="alpha_l2",
        fixed_var_name="scale_l2", fixed_var_value=0.0,
        baseline_nll=inputs["baseline_test_nll"],
        out_path=OUT_DIR / "alpha_l2_grid.png",
        title="Sweep по alpha_l2 (scale_l2 = 0)",
    )

    # Sweep B: scale_l2 grid, alpha_l2 fixed at 0
    print("\n=== Sweep B: scale_l2 (alpha_l2 = 0) ===")
    rows_b = run_sweep(
        inputs,
        alpha_l2_values=[0.0] * len(SCALE_L2_GRID),
        scale_l2_values=SCALE_L2_GRID,
        label="B",
    )
    pd.DataFrame(rows_b).to_csv(OUT_DIR / "sweep_scale_l2.csv", index=False)
    plot_grid(
        rows_b, sweep_var="scale_l2",
        fixed_var_name="alpha_l2", fixed_var_value=0.0,
        baseline_nll=inputs["baseline_test_nll"],
        out_path=OUT_DIR / "scale_l2_grid.png",
        title="Sweep по scale_l2 (alpha_l2 = 0)",
    )

    summary = {
        "personalized_baseline_test_nll": inputs["baseline_test_nll"],
        "sweep_alpha_l2": rows_a,
        "sweep_scale_l2": rows_b,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved CSVs and plots to {OUT_DIR}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

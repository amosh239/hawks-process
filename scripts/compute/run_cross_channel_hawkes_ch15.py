"""Chapter 15: cross-channel Scaled-baseline Hawkes.

For each of K behavioural channels (default: 3 — `searches`, `to_cart`,
`to_ord`) we treat the channel as the target and fit a separate
Scaled-baseline Hawkes:

    λ_i[u, t] = c_i · μ_u^{EB,i} · b_t^i + Σ_j α_{i,j} · z_j[u, t]

where `z_j` are exp-decay states with a single half-life of 1 day for all
K source channels. The result is a K-vector `c` and a K×K matrix `α`
that captures pairwise short-memory excitation across channels.

Train protocol matches ch.6: 207 days train (2025-01-15 .. 2025-08-09),
52 days test (2025-08-10 .. 2025-09-30), `alpha_l2 = 1e-4`, `scale_l2 = 10.0`.

The 5-channel screening run that motivates dropping `cat_to_cart` and
`cat_to_ord` lives under `screening_5ch/` and is invoked with
`--channels searches,cat_to_cart,cat_to_ord,to_cart,to_ord
 --output-dir diploma/reports/15_cross_channel_hawkes/screening_5ch`.

Default artifacts (3-channel main run):
  diploma/reports/15_cross_channel_hawkes/main_3ch/c_vector.csv
  diploma/reports/15_cross_channel_hawkes/main_3ch/alpha_matrix.csv
  diploma/reports/15_cross_channel_hawkes/main_3ch/alpha_heatmap.png
  diploma/reports/15_cross_channel_hawkes/main_3ch/c_vector.png
  diploma/reports/15_cross_channel_hawkes/main_3ch/summary.json
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

from src.diploma_baselines.data import filter_date_range, load_daily_grid
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_user_states_cache,
    fit_pooled_additive_multi_kernel_hawkes,
    predict_pooled_additive_multi_kernel_hawkes,
)


DEFAULT_CHANNELS = ("searches", "to_cart", "to_ord")
SCREENING_CHANNELS = ("searches", "cat_to_cart", "cat_to_ord", "to_cart", "to_ord")
HALF_LIVES = (1.0,)  # single 1-day half-life as requested
WINDOW_SIZE = 7

TRAIN_START = pd.Timestamp("2025-01-15")
TRAIN_END = pd.Timestamp("2025-08-09")
TEST_START = pd.Timestamp("2025-08-10")
TEST_END = pd.Timestamp("2025-09-30")

DEFAULT_OUT_DIR = Path("diploma/reports/15_cross_channel_hawkes/main_3ch")


def fit_one_channel(
    target: str,
    channels: tuple[str, ...],
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    daily_mean_full_per_channel: dict[str, pd.Series],
    states_train: np.ndarray,
    states_test: np.ndarray,
):
    """Fit Scaled-baseline Hawkes with target = `target` channel.

    Returns dict with c, alpha (shape n_features), test NLL, baseline NLL.
    """
    y_train = train_df[target].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    train_daily_mean = train_df.groupby("event_date")[target].mean().sort_index()
    rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean, daily_mean_full_per_channel[target]
    )
    base_train = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    base_test = rs.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)

    train_uids = train_df["user_id"].to_numpy()
    test_uids = test_df["user_id"].to_numpy()
    scaler = PersonalizedGammaPoissonScaler().fit(train_uids, y_train, base_train)
    pers_train = scaler.predict(train_uids, base_train, method="posterior_mean")
    pers_test = scaler.predict(test_uids, base_test, method="posterior_mean")

    baseline_test_nll = float(evaluate_count_forecast(y_test, pers_test)["mean_poisson_nll"])

    train_groups = train_df.groupby("user_id", sort=False).indices
    test_groups = test_df.groupby("user_id", sort=False).indices
    train_state_blocks = []
    train_y_blocks = []
    train_base_blocks = []
    for uid, idx in train_groups.items():
        train_state_blocks.append(states_train[idx])
        train_y_blocks.append(y_train[idx])
        train_base_blocks.append(pers_train[idx])

    hawkes = fit_pooled_additive_multi_kernel_hawkes(
        state_blocks=train_state_blocks,
        y_blocks=train_y_blocks,
        base_blocks=train_base_blocks,
        half_lives=HALF_LIVES,
        feature_names=channels,
        alpha_l2=1e-4,
        learn_base_scale=True,
        scale_l2=10.0,
        scale_init=1.0,
        max_iter=300,
    )

    # Train NLL of Hawkes prediction
    train_pred = np.clip(
        hawkes.base_scale * pers_train + states_train @ hawkes.alpha, 1e-8, None,
    )
    hawkes_train_nll = float(evaluate_count_forecast(y_train, train_pred)["mean_poisson_nll"])
    baseline_train_nll = float(evaluate_count_forecast(y_train, pers_train)["mean_poisson_nll"])

    test_state_blocks = []
    test_y_blocks = []
    test_base_blocks = []
    for uid, idx in test_groups.items():
        test_state_blocks.append(states_test[idx])
        test_y_blocks.append(y_test[idx])
        test_base_blocks.append(pers_test[idx])
    test_preds = []
    for sb, bb in zip(test_state_blocks, test_base_blocks):
        lam, _ = predict_pooled_additive_multi_kernel_hawkes(
            hawkes, states=sb, base_lambda=bb,
        )
        test_preds.append(lam)
    test_pred_concat = np.concatenate(test_preds)
    # Reconstruct test_y order to match concat order
    test_y_concat = np.concatenate(test_y_blocks)
    hawkes_test_nll = float(evaluate_count_forecast(test_y_concat, test_pred_concat)["mean_poisson_nll"])

    return {
        "target": target,
        "c": float(hawkes.base_scale),
        "alpha": np.asarray(hawkes.alpha, dtype=float),  # shape (n_features,) since 1 half-life
        "alpha_norm": float(np.linalg.norm(hawkes.alpha)),
        "baseline_train_nll": baseline_train_nll,
        "hawkes_train_nll": hawkes_train_nll,
        "baseline_test_nll": baseline_test_nll,
        "hawkes_test_nll": hawkes_test_nll,
        "delta_nll_train": hawkes_train_nll - baseline_train_nll,
        "delta_nll_test": hawkes_test_nll - baseline_test_nll,
        "converged": bool(hawkes.success),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-channel Hawkes (chapter 15)")
    parser.add_argument(
        "--channels",
        type=str,
        default=",".join(DEFAULT_CHANNELS),
        help="Comma-separated channel list (target = source set).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Where to write artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Channels ({len(channels)}): {channels}")
    print(f"Output dir: {out_dir}")

    print("Loading data...")
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=list(channels),
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = {ch: full_df.groupby("event_date")[ch].mean().sort_index() for ch in channels}

    train_df = filter_date_range(full_df, start_date=TRAIN_START, end_date=TRAIN_END).copy()
    test_df = filter_date_range(full_df, start_date=TEST_START, end_date=TEST_END).copy()
    print(f"  train rows: {len(train_df):,}, test rows: {len(test_df):,}")

    print(f"\nBuilding Hawkes states cache ({len(channels)} channels × 1 half-life = {len(channels)} features)...")
    cache = build_user_states_cache(full_df, features=channels, half_lives=HALF_LIVES)
    states_train = cache.gather_for(train_df)
    states_test = cache.gather_for(test_df)
    print(f"  states_train shape = {states_train.shape}")

    rows = []
    alpha_matrix = np.zeros((len(channels), len(channels)), dtype=float)
    c_vector = np.zeros(len(channels), dtype=float)

    for i, target in enumerate(channels):
        print(f"\n=== fitting target = {target!r} ===")
        t0 = time.time()
        res = fit_one_channel(
            target=target,
            channels=channels,
            full_df=full_df,
            train_df=train_df,
            test_df=test_df,
            daily_mean_full_per_channel=daily_mean_full,
            states_train=states_train,
            states_test=states_test,
        )
        elapsed = time.time() - t0
        rows.append(res)
        c_vector[i] = res["c"]
        alpha_matrix[i, :] = res["alpha"]
        print(
            f"  c = {res['c']:.4f}, ||α|| = {res['alpha_norm']:.4f}, "
            f"baseline NLL = {res['baseline_test_nll']:.5f}, "
            f"hawkes NLL = {res['hawkes_test_nll']:.5f}, "
            f"Δ = {res['delta_nll_test']:+.5f}  ({elapsed:.1f}s, ok={res['converged']})"
        )
        for j, src in enumerate(channels):
            print(f"    α[{target} <- {src:>12s}] = {res['alpha'][j]:+.4f}")

    print("\n=== c vector ===")
    for i, ch in enumerate(channels):
        print(f"  c[{ch}] = {c_vector[i]:.4f}")

    print("\n=== α matrix (rows = target, cols = source) ===")
    print(f"  {'':<13s} " + "  ".join(f"{src:>12s}" for src in channels))
    for i, target in enumerate(channels):
        print(f"  {target:<13s} " + "  ".join(f"{alpha_matrix[i, j]:+12.4f}" for j in range(len(channels))))

    # === Persist artifacts ===
    pd.DataFrame({"channel": list(channels), "c": c_vector}).to_csv(out_dir / "c_vector.csv", index=False)
    alpha_df = pd.DataFrame(alpha_matrix, index=list(channels), columns=list(channels))
    alpha_df.index.name = "target"
    alpha_df.to_csv(out_dir / "alpha_matrix.csv")

    summary = {
        "train_window": {"start": str(TRAIN_START.date()), "end": str(TRAIN_END.date())},
        "test_window": {"start": str(TEST_START.date()), "end": str(TEST_END.date())},
        "half_lives": list(HALF_LIVES),
        "channels": list(channels),
        "c": dict(zip(channels, c_vector.tolist())),
        "alpha": {
            target: dict(zip(channels, alpha_matrix[i].tolist()))
            for i, target in enumerate(channels)
        },
        "per_channel_metrics": [
            {
                "target": r["target"],
                "c": r["c"],
                "alpha_norm": r["alpha_norm"],
                "baseline_test_nll": r["baseline_test_nll"],
                "hawkes_test_nll": r["hawkes_test_nll"],
                "delta_nll_test": r["delta_nll_test"],
                "converged": r["converged"],
            }
            for r in rows
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # === Heatmap ===
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    vmax = float(np.max(np.abs(alpha_matrix)))
    im = ax.imshow(alpha_matrix, cmap="RdBu_r", vmin=-vmax, vmax=+vmax, aspect="auto")
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=35, ha="right")
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_xlabel("Source channel  (j)")
    ax.set_ylabel("Target channel  (i)")
    ax.set_title(r"Cross-channel Hawkes $\alpha_{i,j}$  (half-life = 1 day, 207d train)")
    for i in range(len(channels)):
        for j in range(len(channels)):
            val = alpha_matrix[i, j]
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center", fontsize=10,
                color="white" if abs(val) > 0.5 * vmax else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\alpha_{i,j}$")
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_heatmap.png", dpi=150)
    plt.close(fig)

    # === c-vector bar plot ===
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(range(len(channels)), c_vector, color="#2E5EAA", edgecolor="white")
    ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle="--", label="c = 1 (no rescale)")
    for i, val in enumerate(c_vector):
        ax.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=10, color="#0B3C5D", fontweight="bold")
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=20, ha="right")
    ax.set_ylabel("c (baseline rescale)")
    ax.set_title("Cross-channel Hawkes: per-channel baseline scale c")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "c_vector.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved artifacts to {out_dir}")


if __name__ == "__main__":
    main()

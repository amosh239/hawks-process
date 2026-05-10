"""Per-user multiplier scatter plot for chapter 10.4.1.

For 2 randomly chosen blocks (seed=42):
  x-axis = μ_u from Personalized Gamma-Poisson (EB posterior mean)
  y-axis = λ_u from Personalized L2 (joint MLE with γ=1)
Each point is one user (10K points per panel).

If both regularizers gave the same per-user multiplier, all points would lie
on the diagonal. Deviation from the diagonal shows where Gamma-EB vs L2
disagree.
"""

from __future__ import annotations

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
from scipy.optimize import minimize

from src.diploma_baselines.data import load_daily_grid
from src.diploma_baselines.models import (
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
)


TARGET_COL = "to_ord"
WINDOW_SIZE = 7

CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14
GAMMA_L2 = 1.0  # same as Personalized L2

OUTPUT_DIR = Path("diploma/reports/blockwise_cv")


def fit_personalized_l2(user_idx_arr, y, b, n_users, gamma):
    init_lam = np.ones(n_users)
    bounds = [(0.001, 50.0)] * n_users

    def _obj(params):
        mu = np.clip(params[user_idx_arr] * b, 1e-8, None)
        nll = float(
            np.sum(mu - y * np.log(mu))
            + gamma * np.sum((params - 1.0) ** 2)
        )
        residual = 1.0 - y / mu
        grad = np.zeros(n_users)
        np.add.at(grad, user_idx_arr, residual * b)
        grad += 2.0 * gamma * (params - 1.0)
        return nll, grad

    res = minimize(_obj, init_lam, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 400})
    return res.x


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=[TARGET_COL],
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    blocks = []
    cursor = CV_GLOBAL_START
    idx = 0
    while True:
        block_start = cursor
        block_end = cursor + pd.Timedelta(days=BLOCK_LEN - 1)
        if block_end > CV_GLOBAL_END:
            break
        train_end = block_start + pd.Timedelta(days=TRAIN_LEN - 1)
        blocks.append({
            "block_idx": idx,
            "block_start": block_start,
            "train_end": train_end,
        })
        idx += 1
        cursor = block_end + pd.Timedelta(days=1)

    # Pick 2 random blocks reproducibly
    rng = np.random.default_rng(42)
    chosen = sorted(rng.choice(len(blocks), size=2, replace=False).tolist())
    print(f"\nChosen blocks: {[b + 1 for b in chosen]} (B{chosen[0] + 1}, B{chosen[1] + 1})")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.5), sharex=True, sharey=True)

    for panel_idx, block_idx in enumerate(chosen):
        block = blocks[block_idx]
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
        train_uids = block_train_df["user_id"].to_numpy()
        y_train = block_train_df[TARGET_COL].to_numpy(dtype=float)

        # Personalized Gamma-Poisson (EB)
        scaler = PersonalizedGammaPoissonScaler().fit(train_uids, y_train, base_train)
        mu_per_user_series = scaler.user_stats_["mu_posterior_mean"]
        # Stats indexed by user_id
        eb_alpha, eb_beta = scaler.alpha_, scaler.beta_
        prior_mean = scaler.prior_mean_

        # Personalized L2 (joint MLE λ-only)
        unique_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
        n_users = int(len(unique_uids))
        lam_u_l2 = fit_personalized_l2(train_user_idx, y_train, base_train, n_users, GAMMA_L2)

        # Match: for each user in unique_uids, get μ_u from scaler
        mu_per_user_arr = np.array(
            [float(mu_per_user_series.get(int(uid), prior_mean)) for uid in unique_uids],
            dtype=float,
        )

        # Per-user event count for color
        y_per_user = np.zeros(n_users, dtype=float)
        np.add.at(y_per_user, train_user_idx, y_train)

        ax = axes[panel_idx]

        # Diagonal y=x
        max_val = max(np.percentile(mu_per_user_arr, 99.5), np.percentile(lam_u_l2, 99.5)) * 1.05
        max_val = min(max_val, 8.0)
        ax.plot([0, max_val], [0, max_val], color="#888888", linewidth=1.0, linestyle="--", zorder=1, label="y = x (одинаковая регуляризация)")
        ax.axhline(1.0, color="#aaaaaa", linewidth=0.6, linestyle=":", zorder=0)
        ax.axvline(prior_mean, color="#aaaaaa", linewidth=0.6, linestyle=":", zorder=0)

        # Color points by user activity
        sc = ax.scatter(
            mu_per_user_arr, lam_u_l2,
            c=y_per_user, cmap="viridis",
            s=8, alpha=0.55, edgecolors="none",
            vmin=0, vmax=np.percentile(y_per_user, 99),
        )

        # Means
        mean_mu = float(np.mean(mu_per_user_arr))
        mean_lam = float(np.mean(lam_u_l2))
        ax.scatter([mean_mu], [mean_lam], color="red", marker="x", s=80, linewidths=2.5, zorder=5,
                   label=f"means: μ̄={mean_mu:.3f}, λ̄={mean_lam:.3f}")

        ax.set_xlabel("μ_u  (Personalized Gamma-Poisson, EB posterior mean)")
        if panel_idx == 0:
            ax.set_ylabel("λ_u  (Personalized L2, joint MLE с γ=1)")
        ax.set_title(
            f"Block B{block_idx + 1}: {block_start.date()}..{train_end.date()}  "
            f"(EB α₀={eb_alpha:.3f}, β₀={eb_beta:.3f}, prior_mean={prior_mean:.3f})",
            fontsize=10,
        )
        ax.set_xlim(-0.05, max_val)
        ax.set_ylim(-0.05, max_val)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=9)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("Σ событий юзера на 14d train")

    fig.suptitle(
        "Per-user multiplier на season_poisson: Personalized Gamma-Poisson (oX) vs Personalized L2 (oY) на 2 блоках",
        fontsize=13,
    )
    fig.savefig(OUTPUT_DIR / "per_user_multipliers_scatter.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved scatter plot to {OUTPUT_DIR / 'per_user_multipliers_scatter.png'}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

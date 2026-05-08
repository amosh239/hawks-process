"""Generate per-user multiplier scatter for chapter 8 section 8.6.

For each train user computes:
  X = m_staged = c · μ_u^EB        (Scaled-baseline Hawkes from chapter 6)
  Y = m_joint  = λ_u (joint γ=1)   (Joint Hawkes)

Saves:
  diploma/reports/ladder_summary/multiplier_scatter.png
  diploma/reports/ladder_summary/multiplier_scatter.csv
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
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_basis_states,
)
from scripts.run_joint_lambda_alpha_fit import fit_joint  # type: ignore


HAWKES_FEATURES = tuple(FEATURE_NAMES)
TARGET_COL = "to_ord"
HALF_LIVES = (1.0, 3.0)
ANALYSIS_START = pd.Timestamp("2025-01-15")
ANALYSIS_END = pd.Timestamp("2025-09-30")
TRAIN_RATIO = 0.8
WINDOW_SIZE = 7

OUT_DIR = Path("diploma/reports/ladder_summary")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    analysis_df = filter_date_range(full_df, start_date=ANALYSIS_START, end_date=ANALYSIS_END)
    split = split_panel_by_date(analysis_df, train_ratio=TRAIN_RATIO)
    train_df = split.train.reset_index(drop=True)

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    train_daily_mean = train_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean, daily_mean_full
    )
    base_train = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)

    # === Personalized GP -> mu_u^EB per user ===
    print("Fitting Personalized GP...")
    scaler = PersonalizedGammaPoissonScaler().fit(
        train_df["user_id"].to_numpy(), y_train, base_train,
    )
    mu_eb = scaler.user_stats_["mu_posterior_mean"].rename("mu_eb").reset_index()
    mu_eb["user_id"] = mu_eb["user_id"].astype(int)
    print(f"  EB params alpha0={scaler.alpha_:.4f}, beta0={scaler.beta_:.4f}")

    # === Staged c from existing summary ===
    staged_summary = json.loads(
        Path("diploma/reports/joint_lambda_alpha/summary.json").read_text()
    )["A_staged_EB_plus_Hawkes"]
    c_staged = float(staged_summary["c"])
    print(f"  staged c (from summary) = {c_staged:.4f}")

    # === Joint Hawkes (γ=1) -> lambda_u per user ===
    print("Building Hawkes states for train...")
    beta = np.log(2.0) / np.asarray(HALF_LIVES, dtype=float)
    n_alpha = len(HAWKES_FEATURES) * len(HALF_LIVES)

    train_uids = train_df["user_id"].to_numpy()
    unique_train_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
    n_users = int(len(unique_train_uids))

    states_train = np.zeros((len(train_df), n_alpha), dtype=np.float32)
    train_dates = train_df["event_date"].to_numpy(dtype="datetime64[ns]")
    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        full_dates = full_user["event_date"].to_numpy(dtype="datetime64[ns]")
        full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}
        groups = train_df.groupby("user_id", sort=False).indices
        if int(user_id) in groups:
            idx = groups[int(user_id)]
            wanted = train_dates[idx]
            rows_in_full = np.array([full_to_idx[pd.Timestamp(d).normalize()] for d in wanted], dtype=int)
            states_train[idx] = states_full[rows_in_full]

    print("Fitting joint Hawkes (γ=1)...")
    lam_u_fit, alpha_fit, info = fit_joint(
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
    print(f"  joint mean λ_u = {lam_u_fit.mean():.4f}, ||α|| = {np.linalg.norm(alpha_fit):.4f}")

    joint_df = pd.DataFrame({"user_id": unique_train_uids.astype(int), "lambda_u": lam_u_fit})

    # === Merge and build per-user table ===
    user_purchases = train_df.groupby("user_id", as_index=False)[TARGET_COL].sum().rename(
        columns={TARGET_COL: "train_purchases", "user_id": "user_id"}
    )
    user_purchases["user_id"] = user_purchases["user_id"].astype(int)

    merged = mu_eb.merge(joint_df, on="user_id", how="inner").merge(user_purchases, on="user_id", how="left")
    merged["m_staged"] = c_staged * merged["mu_eb"]
    merged["m_joint"] = merged["lambda_u"]
    merged.to_csv(OUT_DIR / "multiplier_scatter.csv", index=False)
    print(f"  merged users: {len(merged):,}")

    # === Plot: side-by-side (full + clipped 1% extreme) ===
    mean_staged = float(merged["m_staged"].mean())
    mean_joint = float(merged["m_joint"].mean())

    # Clip 1% of points with the largest max-coordinate
    max_coord = merged[["m_staged", "m_joint"]].max(axis=1)
    q99 = float(max_coord.quantile(0.99))
    clipped = merged[max_coord <= q99].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.8))

    def draw_panel(ax, df, title_suffix, lim):
        sc = ax.scatter(
            df["m_staged"], df["m_joint"],
            c=df["train_purchases"].clip(0, 30),
            s=10, alpha=0.45, cmap="viridis", edgecolors="none",
        )
        ax.plot([0, lim], [0, lim], color="#888888", linestyle="--", linewidth=1.0, label="y = x")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.axhline(mean_joint, color="#D2691E", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.axvline(mean_staged, color="#D2691E", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.scatter([mean_staged], [mean_joint], marker="x", color="#D2691E",
                   s=120, linewidths=2.5, zorder=5,
                   label=f"mean ({mean_staged:.3f}, {mean_joint:.3f})")
        ax.set_xlabel(f"m_u staged Hawkes  =  c · μ_u^EB   (c = {c_staged:.3f})")
        ax.set_ylabel("m_u joint Hawkes  =  λ_u (γ=1)")
        ax.set_title(title_suffix)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.grid(linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        return sc

    # Left: full
    full_lim = float(max(merged["m_staged"].quantile(0.999), merged["m_joint"].quantile(0.999))) * 1.05
    sc_full = draw_panel(axes[0], merged,
                         f"Все юзеры ({len(merged):,})", full_lim)

    # Right: clipped at 99th percentile of max coord
    clipped_lim = q99 * 1.05
    n_dropped = len(merged) - len(clipped)
    sc_clip = draw_panel(axes[1], clipped,
                         f"Без 1% самых крайних ({len(clipped):,} юзеров; {n_dropped:,} убрано)",
                         clipped_lim)

    fig.suptitle("Per-user множитель перед `season_poisson`:  Scaled-baseline (X)  vs  Joint Hawkes (Y)",
                 fontsize=12)
    cbar = fig.colorbar(sc_clip, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("train purchases (clipped at 30)", fontsize=9)
    fig.savefig(OUT_DIR / "multiplier_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_DIR / 'multiplier_scatter.png'}")
    print(f"  mean m_staged={merged['m_staged'].mean():.4f}, mean m_joint={merged['m_joint'].mean():.4f}")
    print(f"  corr(m_staged, m_joint) = {merged['m_staged'].corr(merged['m_joint']):.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

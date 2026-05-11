"""Chapter 16 — per-user intensity traces: Personalized GP vs Joint Hawkes.

Chapter-6 protocol:
  Train: 2025-01-15 .. 2025-08-09   (~207 days)
  Test:  2025-08-10 .. 2025-09-30   (~52 days)

What it does:
  1. Fit Personalized Gamma-Poisson scaler on train.
  2. Fit Joint Hawkes (`λ_l2 = 1`) on train.
  3. Compute per-(user, day) predicted intensity on the full analysis window.
  4. Pick 4 representative users (active / inactive / Joint wins / Joint loses on test).
  5. Save two PNGs (one per model) with vertical bars at purchase days
     and intensity on background.

Outputs (under diploma/reports/24_user_lifecycle/):
  - personalized_gp_traces.png
  - joint_hawkes_traces.png
  - user_traces.csv
  - summary.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
(ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache").mkdir(parents=True, exist_ok=True)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diploma_baselines.data import filter_date_range, load_daily_grid, split_panel_by_date
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_user_states_cache,
    fit_joint_hawkes,
)


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

ANALYSIS_START = pd.Timestamp("2025-01-15")
ANALYSIS_END = pd.Timestamp("2025-09-30")
TRAIN_RATIO = 0.8

LAMBDA_L2 = 1.0
ALPHA_L2 = 1e-4
MAX_ITER = 600

PANEL_PATH = "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv"
OUT_DIR = Path("diploma/reports/24_user_lifecycle")


def per_user_test_ll(test_df: pd.DataFrame, lam_col: str) -> pd.Series:
    """Sum poisson log-likelihood per user on test rows.

    LL_i = y log(λ) - λ - lgamma(y+1).
    """
    y = test_df[TARGET_COL].to_numpy(dtype=float)
    lam = np.clip(test_df[lam_col].to_numpy(dtype=float), 1e-12, None)
    ll = y * np.log(lam) - lam - _lgamma_factorial(y)
    out = pd.Series(ll, index=test_df.index)
    return out.groupby(test_df["user_id"]).sum()


def _lgamma_factorial(y: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln
    return gammaln(np.asarray(y, dtype=float) + 1.0)


def pick_users(stats: pd.DataFrame) -> list[dict]:
    """Pick four users: active, inactive, joint-wins, joint-loses.

    `stats` columns: user_id, total_y_train, total_y_test, ll_pers_gp_test, ll_joint_hawkes_test, delta_ll
    """
    # Active: high train activity AND non-trivial test activity, but pick mid-range
    # (avoid the absolute outlier — those tend to have weird dynamics).
    active_cand = stats[(stats["total_y_train"] >= 50) & (stats["total_y_test"] >= 10)]
    active_cand = active_cand.sort_values("total_y_train", ascending=False).head(50)
    active = active_cand.iloc[len(active_cand) // 3]  # upper-third of top 50

    # Inactive: small but non-zero — visible bars
    inactive_cand = stats[(stats["total_y_train"].between(2, 8)) & (stats["total_y_test"] >= 1)]
    inactive = inactive_cand.sort_values("total_y_test").iloc[len(inactive_cand) // 2]

    # Joint wins on test: largest positive delta, among users active in test
    eligible = stats[stats["total_y_test"] >= 3]
    joint_wins = eligible.sort_values("delta_ll", ascending=False).iloc[0]
    joint_loses = eligible.sort_values("delta_ll", ascending=True).iloc[0]

    out = []
    for kind, row in [
        ("активный", active),
        ("малоактивный", inactive),
        ("Joint Hawkes выигрывает", joint_wins),
        ("Joint Hawkes проигрывает", joint_loses),
    ]:
        out.append({
            "user_id": int(row["user_id"]),
            "kind": kind,
            "total_y_train": int(row["total_y_train"]),
            "total_y_test": int(row["total_y_test"]),
            "ll_pers_gp_test": float(row["ll_pers_gp_test"]),
            "ll_joint_hawkes_test": float(row["ll_joint_hawkes_test"]),
            "delta_ll": float(row["delta_ll"]),
        })
    return out


def plot_traces(traces: pd.DataFrame, users: list[dict], lam_col: str, model_name: str,
                color: str, split_date: pd.Timestamp, out_path: Path) -> None:
    n = len(users)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, info in zip(axes, users):
        uid = info["user_id"]
        df = traces[traces["user_id"] == uid].sort_values("event_date")
        dates = df["event_date"].to_numpy()
        y = df[TARGET_COL].to_numpy(dtype=float)
        lam = df[lam_col].to_numpy(dtype=float)

        # Intensity background
        ax.fill_between(dates, 0, lam, color=color, alpha=0.30, linewidth=0)
        ax.plot(dates, lam, color=color, lw=1.2, alpha=0.95)

        # Purchase vertical bars
        purchases = y > 0
        if purchases.any():
            ax.vlines(dates[purchases], ymin=0, ymax=y[purchases],
                      color="black", lw=1.4, alpha=0.85, zorder=4)

        # Train/test boundary
        ax.axvline(split_date, color="#A00", linestyle="--", linewidth=1.0, alpha=0.7)

        y_top = max(float(y.max()) if y.size else 0.0, float(lam.max()) if lam.size else 0.0)
        ax.set_ylim(0, y_top * 1.10 + 1e-3)
        ax.set_ylabel("count / λ̂", fontsize=9)
        ax.set_title(
            f"user {uid} — {info['kind']}: "
            f"train Y={info['total_y_train']}, test Y={info['total_y_test']}, "
            f"ΔLL(JH−GP) on test = {info['delta_ll']:+.2f}",
            fontsize=10,
        )
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("date")

    fig.suptitle(
        f"{model_name}: предсказанная дневная интенсивность и фактические покупки",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading panel...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(PANEL_PATH, value_cols=cols)
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    analysis_df = filter_date_range(full_df, start_date=ANALYSIS_START, end_date=ANALYSIS_END)
    split = split_panel_by_date(analysis_df, train_ratio=TRAIN_RATIO)
    split_date = pd.Timestamp(split.split_date)
    print(f"  split_date = {split_date.date()}, train rows = {len(split.train):,}, test rows = {len(split.test):,}")

    # === Baseline b_t ===
    train_daily_mean = split.train.groupby("event_date")[TARGET_COL].mean().sort_index()
    rs_model = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean, daily_mean_full
    )

    analysis_df = analysis_df.copy()
    analysis_df["b_t"] = rs_model.predict_for_dates(analysis_df["event_date"]).to_numpy(dtype=float)

    base_train = analysis_df.loc[split.train.index, "b_t"].to_numpy(dtype=float)

    # === Personalized GP fit ===
    print("\nFitting Personalized GP...")
    train_uids = split.train["user_id"].to_numpy()
    y_train = split.train[TARGET_COL].to_numpy(dtype=float)
    scaler = PersonalizedGammaPoissonScaler().fit(train_uids, y_train, base_train)
    print(f"  alpha={scaler.alpha_:.3f}, beta={scaler.beta_:.3f}")

    # Per-row predicted intensity on full analysis window
    all_uids = analysis_df["user_id"].to_numpy()
    base_full = analysis_df["b_t"].to_numpy(dtype=float)
    analysis_df["lam_pers_gp"] = scaler.predict(all_uids, base_full, method="posterior_mean")

    # === Joint Hawkes fit ===
    print("\nBuilding Hawkes states...")
    cache = build_user_states_cache(full_df, features=HAWKES_FEATURES, half_lives=HALF_LIVES)
    states_full = cache.gather_for(analysis_df).astype(float)
    states_train = cache.gather_for(split.train).astype(float)
    print(f"  states_full shape = {states_full.shape}")

    unique_train_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
    n_users = int(len(unique_train_uids))
    uid_to_idx = {int(u): i for i, u in enumerate(unique_train_uids)}
    full_user_idx = np.array([uid_to_idx.get(int(u), -1) for u in all_uids], dtype=np.int64)

    print(f"\nFitting Joint Hawkes (λ_l2={LAMBDA_L2}, α_l2={ALPHA_L2})...")
    jh = fit_joint_hawkes(
        user_idx=train_user_idx,
        y=y_train,
        b=base_train,
        states=states_train,
        n_users=n_users,
        lambda_l2=LAMBDA_L2,
        alpha_l2=ALPHA_L2,
        max_iter=MAX_ITER,
    )
    print(f"  converged={jh.converged}, n_iter={jh.n_iter}, ‖α‖={np.linalg.norm(jh.alpha):.4f}")

    lam_u_for_full = np.where(full_user_idx >= 0, jh.lam_u[np.maximum(full_user_idx, 0)], 1.0)
    analysis_df["lam_joint_hawkes"] = lam_u_for_full * base_full + states_full @ jh.alpha

    # === Per-user test LL & user selection ===
    print("\nComputing per-user test LL...")
    test_mask = analysis_df["event_date"] >= split_date
    test_df = analysis_df[test_mask]
    train_df = analysis_df[~test_mask]

    ll_gp = per_user_test_ll(test_df, "lam_pers_gp")
    ll_jh = per_user_test_ll(test_df, "lam_joint_hawkes")
    delta_ll = (ll_jh - ll_gp).rename("delta_ll")

    train_y_sum = train_df.groupby("user_id")[TARGET_COL].sum()
    test_y_sum = test_df.groupby("user_id")[TARGET_COL].sum()

    stats = pd.DataFrame({
        "total_y_train": train_y_sum,
        "total_y_test": test_y_sum,
        "ll_pers_gp_test": ll_gp,
        "ll_joint_hawkes_test": ll_jh,
        "delta_ll": delta_ll,
    })
    stats = stats.fillna(0).reset_index()

    users = pick_users(stats)
    print("\nPicked users:")
    for u in users:
        print(f"  {u['kind']:>26s}  user_id={u['user_id']:>8d}  Y_train={u['total_y_train']:>3d}  "
              f"Y_test={u['total_y_test']:>3d}  ΔLL={u['delta_ll']:+.2f}")

    # === Traces dataframe (only picked users, keep both intensities) ===
    picked_ids = [u["user_id"] for u in users]
    traces = analysis_df[analysis_df["user_id"].isin(picked_ids)][
        ["user_id", "event_date", TARGET_COL, "b_t", "lam_pers_gp", "lam_joint_hawkes"]
    ].sort_values(["user_id", "event_date"]).reset_index(drop=True)
    traces.to_csv(OUT_DIR / "user_traces.csv", index=False)

    # === Plots ===
    print("\nRendering figures...")
    plot_traces(
        traces, users, "lam_pers_gp", "Personalized Gamma-Poisson",
        color="#2E5EAA", split_date=split_date,
        out_path=OUT_DIR / "personalized_gp_traces.png",
    )
    plot_traces(
        traces, users, "lam_joint_hawkes", "Joint Hawkes (λ_l2 = 1)",
        color="#7B3FAA", split_date=split_date,
        out_path=OUT_DIR / "joint_hawkes_traces.png",
    )

    # === Summary ===
    summary = {
        "panel_path": PANEL_PATH,
        "target_col": TARGET_COL,
        "analysis_window": [str(ANALYSIS_START.date()), str(ANALYSIS_END.date())],
        "split_date": str(split_date.date()),
        "n_users_train": n_users,
        "n_rows_train": int(len(split.train)),
        "n_rows_test": int(len(split.test)),
        "joint_hawkes_fit": {
            "lambda_l2": LAMBDA_L2,
            "alpha_l2": ALPHA_L2,
            "alpha": jh.alpha.tolist(),
            "alpha_norm": float(np.linalg.norm(jh.alpha)),
            "lam_u_mean": float(np.mean(jh.lam_u)),
            "converged": bool(jh.converged),
            "n_iter": int(jh.n_iter),
        },
        "personalized_gp_fit": {
            "alpha": float(scaler.alpha_),
            "beta": float(scaler.beta_),
        },
        "picked_users": users,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()

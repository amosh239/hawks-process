"""Joint Hawkes (λ_u + α fit jointly) on ch.6 protocol — chapter-8 runner.

Mirrors scripts/run_pooled_hawkes_ch7.py but for the model

  λ_{u,t} = λ_u · b_t + α^T s_{u,t}

with joint Poisson-MLE training and L2 shrinkage (λ_u − 1)² (γ=1) plus L2 on α.

Produces all artifacts used in chapter 8:

  diploma/reports/08_joint_hawkes/summary.json
  diploma/reports/08_joint_hawkes/alpha_heatmap.png
  diploma/reports/08_joint_hawkes/alpha_table.csv
  diploma/reports/08_joint_hawkes/delta_ll_vs_test_purchases.png            (vs Personalized GP)
  diploma/reports/08_joint_hawkes/delta_ll_vs_test_purchases_vs_ch6.png    (vs Scaled-baseline Hawkes)
  diploma/reports/08_joint_hawkes/user_ll_scores.csv
  diploma/reports/08_joint_hawkes/user_ll_scores_vs_ch6.csv
"""

from __future__ import annotations

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
from scipy.special import gammaln

from src.diploma_baselines.data import filter_date_range, load_daily_grid, split_panel_by_date
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_user_states_cache,
)
from scripts.compute.run_joint_lambda_alpha_fit import fit_joint  # type: ignore


HAWKES_FEATURES = tuple(FEATURE_NAMES)
TARGET_COL = "to_ord"
HALF_LIVES = (1.0, 3.0)
ANALYSIS_START = pd.Timestamp("2025-01-15")
ANALYSIS_END = pd.Timestamp("2025-09-30")
TRAIN_RATIO = 0.8
WINDOW_SIZE = 7

ALPHA_L2 = 1e-4
LAMBDA_L2 = 1.0
MAX_ITER = 400

OUT_DIR = Path("diploma/reports/08_joint_hawkes")


def per_user_loglik(user_ids, y, lam):
    lam = np.clip(lam, 1e-8, None)
    log_p = y * np.log(lam) - lam - gammaln(y + 1.0)
    return pd.DataFrame({"user_id": user_ids, "ll": log_p}).groupby("user_id", as_index=False)["ll"].sum()


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
    test_df = split.test.reset_index(drop=True)
    print(f"  train rows: {len(train_df):,}, test rows: {len(test_df):,}")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    train_daily_mean = train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean, daily_mean_full
    )
    base_train = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    base_test = rs.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    # Personalized GP for comparison and per-user LL
    print("Fitting Personalized GP...")
    scaler = PersonalizedGammaPoissonScaler().fit(
        train_df["user_id"].to_numpy(), y_train, base_train,
    )
    pers_test = scaler.predict(test_df["user_id"].to_numpy(), base_test, method="posterior_mean")
    pers_metrics = evaluate_count_forecast(y_test, pers_test)
    print(f"  Personalized GP test NLL = {pers_metrics['mean_poisson_nll']:.5f}")

    # Hawkes states
    print("Building Hawkes states...")
    cache = build_user_states_cache(full_df, features=HAWKES_FEATURES, half_lives=HALF_LIVES)
    n_alpha = cache.n_alpha
    X_train = cache.gather_for(train_df)
    X_test = cache.gather_for(test_df)

    # User indexing
    train_uids = train_df["user_id"].to_numpy()
    unique_train_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
    n_users = int(len(unique_train_uids))
    uid_to_idx = {int(u): i for i, u in enumerate(unique_train_uids)}
    test_user_idx = np.array([uid_to_idx.get(int(u), -1) for u in test_df["user_id"].to_numpy()], dtype=np.int64)

    print(f"\nFitting Joint Hawkes (λ_l2={LAMBDA_L2}, α_l2={ALPHA_L2})...")
    t0 = time.time()
    lam_u_fit, alpha_fit, info = fit_joint(
        user_idx=train_user_idx, y=y_train, b=base_train,
        states=X_train.astype(float),
        n_users=n_users, n_alpha=n_alpha,
        lambda_l2=LAMBDA_L2, alpha_l2=ALPHA_L2,
        max_iter=MAX_ITER, verbose=False,
    )
    print(f"  done in {time.time()-t0:.1f}s. converged={info['converged']}")
    print(f"  mean λ_u = {lam_u_fit.mean():.4f}, ||α||_2 = {np.linalg.norm(alpha_fit):.4f}")

    # Predictions
    lam_u_train_per_row = lam_u_fit[train_user_idx]
    pred_train = np.clip(lam_u_train_per_row * base_train + X_train.astype(float) @ alpha_fit, 1e-8, None)
    train_metrics = evaluate_count_forecast(y_train, pred_train)

    lam_u_for_test = np.where(test_user_idx >= 0, lam_u_fit[np.maximum(test_user_idx, 0)], 1.0)
    pred_test = np.clip(lam_u_for_test * base_test + X_test.astype(float) @ alpha_fit, 1e-8, None)
    test_metrics = evaluate_count_forecast(y_test, pred_test)

    print(f"  train NLL = {train_metrics['mean_poisson_nll']:.5f}")
    print(f"  test  NLL = {test_metrics['mean_poisson_nll']:.5f}")
    print(f"  Δ vs Personalized GP = {test_metrics['poisson_loglik'] - pers_metrics['poisson_loglik']:+.0f} нат")

    # Alpha heatmap
    alpha_matrix = alpha_fit.reshape(len(HAWKES_FEATURES), len(HALF_LIVES))
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    im = ax.imshow(alpha_matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0.0, vmax=float(alpha_matrix.max() * 1.05) + 1e-9)
    ax.set_xticks(range(len(HALF_LIVES)))
    ax.set_xticklabels([str(int(h)) for h in HALF_LIVES])
    ax.set_yticks(range(len(HAWKES_FEATURES)))
    ax.set_yticklabels(HAWKES_FEATURES)
    for i in range(alpha_matrix.shape[0]):
        for j in range(alpha_matrix.shape[1]):
            ax.text(j, i, f"{alpha_matrix[i, j]:.4f}", ha="center", va="center",
                    fontsize=9, color="#111111")
    ax.set_xlabel("half-life (days)")
    ax.set_ylabel("feature")
    ax.set_title(f"Joint Hawkes α-matrix (mean λ_u = {lam_u_fit.mean():.4f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_heatmap.png", dpi=150)
    plt.close(fig)

    # Alpha table
    rows = []
    for fi, fname in enumerate(HAWKES_FEATURES):
        for hi, hl in enumerate(HALF_LIVES):
            rows.append({"feature": fname, "half_life": int(hl), "alpha": float(alpha_matrix[fi, hi])})
    pd.DataFrame(rows).to_csv(OUT_DIR / "alpha_table.csv", index=False)

    # Per-user LL: Joint vs Personalized GP
    joint_user_ll = per_user_loglik(test_df["user_id"].to_numpy(), y_test, pred_test)
    pers_user_ll = per_user_loglik(test_df["user_id"].to_numpy(), y_test, pers_test)
    user_purchases = test_df.groupby("user_id", as_index=False)[TARGET_COL].sum().rename(
        columns={TARGET_COL: "test_purchases"}
    )
    merged_pers = (
        joint_user_ll.rename(columns={"ll": "ll_joint"})
        .merge(pers_user_ll.rename(columns={"ll": "ll_pers"}), on="user_id")
        .merge(user_purchases, on="user_id")
    )
    merged_pers["delta_ll"] = merged_pers["ll_joint"] - merged_pers["ll_pers"]
    merged_pers.to_csv(OUT_DIR / "user_ll_scores.csv", index=False)

    # Per-user LL: Joint vs Scaled-baseline Hawkes (ch.6)
    ch6_user_ll = pd.read_csv("diploma/reports/experimental_1_hawkes/user_ll_scores.csv")
    merged_ch6 = (
        joint_user_ll.rename(columns={"ll": "ll_joint"})
        .merge(ch6_user_ll[["user_id", "ll_experimental_1_hawkes", "test_purchases"]], on="user_id")
    )
    merged_ch6["delta_ll"] = merged_ch6["ll_joint"] - merged_ch6["ll_experimental_1_hawkes"]
    merged_ch6.to_csv(OUT_DIR / "user_ll_scores_vs_ch6.csv", index=False)

    # Bucket summary helper
    def bucket(p):
        if p == 0: return "0"
        if p == 1: return "1"
        if p == 2: return "2"
        if p <= 5: return "3-5"
        if p <= 10: return "6-10"
        return "11+"
    BUCKETS = ["0", "1", "2", "3-5", "6-10", "11+"]

    def summarize(df):
        df = df.copy()
        df["bucket"] = df["test_purchases"].map(bucket)
        out = df.groupby("bucket").agg(
            n=("delta_ll", "size"),
            mean_delta=("delta_ll", "mean"),
            share_pos=("delta_ll", lambda v: float((v > 0).mean())),
        ).reindex(BUCKETS)
        return out, {
            "share_pos_overall": float((df["delta_ll"] > 0).mean()),
            "mean_delta_ll": float(df["delta_ll"].mean()),
            "median_delta_ll": float(df["delta_ll"].median()),
        }

    pers_buckets, pers_overall = summarize(merged_pers)
    ch6_buckets, ch6_overall = summarize(merged_ch6)

    print("\n=== vs Personalized GP ===")
    print(pers_overall)
    print(pers_buckets)

    print("\n=== vs Scaled-baseline Hawkes (ch.6) ===")
    print(ch6_overall)
    print(ch6_buckets)

    # Summary
    summary = {
        "test_panel": {"rows": int(len(y_test))},
        "test_metrics_joint_hawkes": {
            "poisson_loglik": float(test_metrics["poisson_loglik"]),
            "mean_poisson_nll": float(test_metrics["mean_poisson_nll"]),
            "mean_poisson_deviance": float(test_metrics["mean_poisson_deviance"]),
            "mae": float(test_metrics["mae"]),
            "rmse": float(test_metrics["rmse"]),
            "aggregate_bias": float(test_metrics["aggregate_bias"]),
            "relative_aggregate_bias": float(test_metrics["relative_aggregate_bias"]),
            "mean_target": float(test_metrics["mean_target"]),
            "mean_prediction": float(test_metrics["mean_prediction"]),
        },
        "train_metrics_joint_hawkes": {
            "poisson_loglik": float(train_metrics["poisson_loglik"]),
            "mean_poisson_nll": float(train_metrics["mean_poisson_nll"]),
            "mean_poisson_deviance": float(train_metrics["mean_poisson_deviance"]),
            "mae": float(train_metrics["mae"]),
            "rmse": float(train_metrics["rmse"]),
            "aggregate_bias": float(train_metrics["aggregate_bias"]),
        },
        "fit_params": {
            "lambda_l2": LAMBDA_L2,
            "alpha_l2": ALPHA_L2,
            "alpha_norm": float(np.linalg.norm(alpha_fit)),
            "alpha": alpha_fit.tolist(),
            "lambda_stats": {
                "mean": float(lam_u_fit.mean()),
                "median": float(np.median(lam_u_fit)),
                "std": float(lam_u_fit.std()),
                "p05": float(np.quantile(lam_u_fit, 0.05)),
                "p95": float(np.quantile(lam_u_fit, 0.95)),
                "frac_at_lower_bound": float((lam_u_fit <= 0.001 + 1e-9).mean()),
            },
            "feature_names": list(HAWKES_FEATURES),
            "half_lives": list(HALF_LIVES),
            "n_users": n_users,
            "converged": info["converged"],
        },
        "user_level_vs_personalized_gp": {
            **pers_overall,
            "by_bucket": pers_buckets.reset_index().fillna(0).to_dict(orient="records"),
        },
        "user_level_vs_scaled_baseline_ch6": {
            **ch6_overall,
            "by_bucket": ch6_buckets.reset_index().fillna(0).to_dict(orient="records"),
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nArtifacts saved to {OUT_DIR}/")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

"""Pooled Hawkes (no per-user multiplier) on ch.6 protocol.

Chapter-7 style runner: produces the full set of artifacts used in chapter 7,
mirroring the chapter-6 layout:

  diploma/reports/07_pooled_hawkes/summary.json
  diploma/reports/07_pooled_hawkes/alpha_heatmap.png
  diploma/reports/07_pooled_hawkes/alpha_table.csv
  diploma/reports/07_pooled_hawkes/daily_aggregate_analysis_window.png
  diploma/reports/07_pooled_hawkes/delta_ll_vs_test_purchases.png
  diploma/reports/07_pooled_hawkes/user_ll_scores.csv
  diploma/reports/07_pooled_hawkes/half_life_sweep.csv

Half-life variants tried:
  (1,)
  (1, 3)              ← main, used for plots and chapter-8 ladder
  (1, 3, 7, 21)
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
from scipy.optimize import minimize
from scipy.special import gammaln

from src.diploma_baselines.data import filter_date_range, load_daily_grid, split_panel_by_date
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_basis_states,
)


HAWKES_FEATURES = tuple(FEATURE_NAMES)
TARGET_COL = "to_ord"
ANALYSIS_START = pd.Timestamp("2025-01-15")
ANALYSIS_END = pd.Timestamp("2025-09-30")
TRAIN_RATIO = 0.8
WINDOW_SIZE = 7

ALPHA_L2 = 1e-4
SCALE_L2 = 10.0
MAX_ITER = 500

MAIN_HALF_LIVES = (1.0, 3.0)
HALF_LIFE_VARIANTS = (
    (1.0,),
    (1.0, 3.0),
    (1.0, 3.0, 7.0, 21.0),
)

OUT_DIR = Path("diploma/reports/07_pooled_hawkes")


def fit_pooled_hawkes(X, y, b_raw, alpha_l2=ALPHA_L2, scale_l2=SCALE_L2, max_iter=MAX_ITER):
    n_alpha = X.shape[1]

    def fg(p):
        c = float(p[0]); alpha = np.asarray(p[1:], dtype=float)
        lam = np.clip(c * b_raw + X @ alpha, 1e-8, None)
        nll = float(np.sum(lam - y * np.log(lam))
                    + alpha_l2 * np.sum(alpha**2)
                    + scale_l2 * (c - 1.0) ** 2)
        d = 1.0 - y / lam
        a_grad = X.T @ d + 2.0 * alpha_l2 * alpha
        c_grad = float(np.sum(b_raw * d) + 2.0 * scale_l2 * (c - 1.0))
        return nll, np.concatenate([[c_grad], a_grad])

    init = np.concatenate([[1.0], np.full(n_alpha, 0.01)])
    bounds = [(0.001, 50.0)] + [(0.0, 10.0)] * n_alpha
    res = minimize(lambda p: fg(p)[0], init, method="L-BFGS-B",
                   jac=lambda p: fg(p)[1], bounds=bounds, options={"maxiter": max_iter})
    return float(res.x[0]), np.asarray(res.x[1:], dtype=float), bool(res.success)


def build_states_for_dates(full_df, target_dates_per_row, half_lives, beta):
    n_alpha = len(HAWKES_FEATURES) * len(half_lives)
    out = np.zeros((len(target_dates_per_row), n_alpha), dtype=np.float32)
    return out


def prepare_states(full_df, train_df, test_df, half_lives):
    """Build Hawkes states for train and test rows, given a half_lives tuple."""
    beta = np.log(2.0) / np.asarray(half_lives, dtype=float)
    n_alpha = len(HAWKES_FEATURES) * len(half_lives)

    train_dates = train_df["event_date"].to_numpy(dtype="datetime64[ns]")
    test_dates = test_df["event_date"].to_numpy(dtype="datetime64[ns]")

    X_train = np.zeros((len(train_df), n_alpha), dtype=np.float32)
    X_test = np.zeros((len(test_df), n_alpha), dtype=np.float32)

    train_groups = train_df.groupby("user_id", sort=False).indices
    test_groups = test_df.groupby("user_id", sort=False).indices

    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        full_dates = full_user["event_date"].to_numpy(dtype="datetime64[ns]")
        full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}

        if int(user_id) in train_groups:
            idx_in_block = train_groups[int(user_id)]
            wanted = train_dates[idx_in_block]
            rows_in_full = np.array([full_to_idx[pd.Timestamp(d).normalize()] for d in wanted], dtype=int)
            X_train[idx_in_block] = states_full[rows_in_full]
        if int(user_id) in test_groups:
            idx_in_block = test_groups[int(user_id)]
            wanted = test_dates[idx_in_block]
            rows_in_full = np.array([full_to_idx[pd.Timestamp(d).normalize()] for d in wanted], dtype=int)
            X_test[idx_in_block] = states_full[rows_in_full]

    return X_train.astype(float), X_test.astype(float)


def per_user_loglik(user_ids, y, lam):
    """Per-user Poisson log-likelihood (full, with -log(y!) constant)."""
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
    b_train = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    b_test = rs.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    # Personalized GP for user-level comparison
    scaler = PersonalizedGammaPoissonScaler().fit(
        train_df["user_id"].to_numpy(), y_train, b_train
    )
    pers_test = scaler.predict(test_df["user_id"].to_numpy(), b_test, method="posterior_mean")
    pers_metrics = evaluate_count_forecast(y_test, pers_test)
    print(f"  Personalized GP test NLL = {pers_metrics['mean_poisson_nll']:.5f}")

    # Half-life variants
    half_life_rows = []
    main_alpha = None; main_c = None
    main_pred_train = None; main_pred_test = None
    for half_lives in HALF_LIFE_VARIANTS:
        print(f"\nFitting Pooled Hawkes with half_lives={half_lives}...")
        X_tr, X_te = prepare_states(full_df, train_df, test_df, half_lives)
        c_fit, alpha_fit, ok = fit_pooled_hawkes(X_tr, y_train, b_train)
        lam_test = np.clip(c_fit * b_test + X_te @ alpha_fit, 1e-8, None)
        m = evaluate_count_forecast(y_test, lam_test)
        lam_train = np.clip(c_fit * b_train + X_tr @ alpha_fit, 1e-8, None)
        m_train = evaluate_count_forecast(y_train, lam_train)
        delta_vs_pers = m["poisson_loglik"] - pers_metrics["poisson_loglik"]
        half_life_rows.append({
            "half_lives": str(list(half_lives)),
            "n_alpha_dims": len(HAWKES_FEATURES) * len(half_lives),
            "n_params": 1 + len(HAWKES_FEATURES) * len(half_lives),
            "c": c_fit,
            "alpha_norm": float(np.linalg.norm(alpha_fit)),
            "train_poisson_loglik": float(m_train["poisson_loglik"]),
            "train_mean_poisson_nll": float(m_train["mean_poisson_nll"]),
            "test_poisson_loglik": float(m["poisson_loglik"]),
            "test_mean_poisson_nll": float(m["mean_poisson_nll"]),
            "test_mean_poisson_deviance": float(m["mean_poisson_deviance"]),
            "test_mae": float(m["mae"]),
            "test_rmse": float(m["rmse"]),
            "test_aggregate_bias": float(m["aggregate_bias"]),
            "test_relative_aggregate_bias": float(m["relative_aggregate_bias"]),
            "delta_loglik_vs_personalized": float(delta_vs_pers),
            "converged": bool(ok),
        })
        print(
            f"  c={c_fit:.4f} ||α||={np.linalg.norm(alpha_fit):.4f}  "
            f"test NLL={m['mean_poisson_nll']:.5f} (Δ vs pers = {delta_vs_pers:+.0f} нат)"
        )

        if half_lives == MAIN_HALF_LIVES:
            main_alpha = alpha_fit; main_c = c_fit
            main_pred_train = lam_train; main_pred_test = lam_test
            main_test_metrics = m; main_train_metrics = m_train

    pd.DataFrame(half_life_rows).to_csv(OUT_DIR / "half_life_sweep.csv", index=False)

    # === Alpha heatmap (main) ===
    alpha_matrix = main_alpha.reshape(len(HAWKES_FEATURES), len(MAIN_HALF_LIVES))
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    im = ax.imshow(alpha_matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=float(alpha_matrix.max() * 1.05))
    ax.set_xticks(range(len(MAIN_HALF_LIVES)))
    ax.set_xticklabels([str(int(h)) for h in MAIN_HALF_LIVES])
    ax.set_yticks(range(len(HAWKES_FEATURES)))
    ax.set_yticklabels(HAWKES_FEATURES)
    for i in range(alpha_matrix.shape[0]):
        for j in range(alpha_matrix.shape[1]):
            ax.text(j, i, f"{alpha_matrix[i, j]:.4f}", ha="center", va="center", fontsize=9, color="#111111")
    ax.set_xlabel("half-life (days)")
    ax.set_ylabel("feature")
    ax.set_title(f"Pooled Hawkes α-matrix (c = {main_c:.4f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_heatmap.png", dpi=150)
    plt.close(fig)

    # === Alpha table CSV ===
    alpha_table_rows = []
    for fi, fname in enumerate(HAWKES_FEATURES):
        for hi, hl in enumerate(MAIN_HALF_LIVES):
            alpha_table_rows.append({
                "feature": fname, "half_life": int(hl), "alpha": float(alpha_matrix[fi, hi])
            })
    pd.DataFrame(alpha_table_rows).to_csv(OUT_DIR / "alpha_table.csv", index=False)

    # === Daily aggregate ===
    full_dates = pd.concat([train_df["event_date"], test_df["event_date"]])
    full_y = pd.concat([train_df[TARGET_COL], test_df[TARGET_COL]])
    full_pred = np.concatenate([main_pred_train, main_pred_test])
    daily_df = pd.DataFrame({"date": full_dates.values, "y": full_y.values, "pred": full_pred}).groupby("date").mean()

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.plot(daily_df.index, daily_df["y"], color="#0B3C5D", linewidth=1.4, label="actual mean")
    ax.plot(daily_df.index, daily_df["pred"], color="#D2691E", linewidth=1.4, label="Pooled Hawkes mean")
    test_start = test_df["event_date"].min()
    ax.axvline(test_start, color="#888888", linestyle="--", linewidth=1.0)
    ax.text(test_start, ax.get_ylim()[1] * 0.95, " test → ", ha="left", va="top", fontsize=9, color="#444444")
    ax.set_xlabel("date")
    ax.set_ylabel("mean intensity per user")
    ax.set_title("Pooled Hawkes: дневная средняя интенсивность")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "daily_aggregate_analysis_window.png", dpi=150)
    plt.close(fig)

    # === User-level delta LL vs Personalized GP ===
    pooled_user_ll = per_user_loglik(test_df["user_id"].to_numpy(), y_test, main_pred_test)
    pers_user_ll = per_user_loglik(test_df["user_id"].to_numpy(), y_test, pers_test)
    user_purchases = test_df.groupby("user_id", as_index=False)[TARGET_COL].sum().rename(
        columns={TARGET_COL: "test_purchases"}
    )
    merged = pooled_user_ll.merge(pers_user_ll, on="user_id", suffixes=("_pooled", "_pers")).merge(user_purchases, on="user_id")
    merged["delta_ll"] = merged["ll_pooled"] - merged["ll_pers"]
    merged.to_csv(OUT_DIR / "user_ll_scores.csv", index=False)

    def bucket(p):
        if p == 0: return "0"
        if p == 1: return "1"
        if p == 2: return "2"
        if p <= 5: return "3-5"
        if p <= 10: return "6-10"
        return "11+"
    merged["bucket"] = merged["test_purchases"].map(bucket)
    bucket_order = ["0", "1", "2", "3-5", "6-10", "11+"]
    bucket_summary = merged.groupby("bucket").agg(
        n=("delta_ll", "size"),
        mean_delta=("delta_ll", "mean"),
        share_pos=("delta_ll", lambda v: float((v > 0).mean())),
    ).reindex(bucket_order)

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    rng = np.random.default_rng(0)
    bucket_x = {b: i for i, b in enumerate(bucket_order)}
    for b in bucket_order:
        sub = merged[merged["bucket"] == b]["delta_ll"].to_numpy()
        x_jitter = bucket_x[b] + (rng.random(len(sub)) - 0.5) * 0.32
        ax.scatter(x_jitter, sub, s=8, alpha=0.4, color="#2E5EAA", edgecolors="none")
    means = bucket_summary["mean_delta"].to_numpy()
    for i, m_val in enumerate(means):
        if pd.notna(m_val):
            ax.hlines(m_val, i - 0.32, i + 0.32, color="#D2691E", linewidth=2.4, zorder=4)
            ax.text(i, m_val, f"{m_val:+.3f}", ha="center", va="bottom", fontsize=9, color="#D2691E", fontweight="bold")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(range(len(bucket_order)))
    ax.set_xticklabels(bucket_order)
    ax.set_xlabel("test purchase count")
    ax.set_ylabel("Δ user-LL  (Pooled Hawkes − Personalized GP)")
    ax.set_title("Pooled Hawkes vs Personalized GP: per-user Δ LL по бакетам")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "delta_ll_vs_test_purchases.png", dpi=150)
    plt.close(fig)

    # === Summary ===
    summary = {
        "test_panel": {"rows": int(len(y_test))},
        "test_metrics_pooled_hawkes": {
            "poisson_loglik": float(main_test_metrics["poisson_loglik"]),
            "mean_poisson_nll": float(main_test_metrics["mean_poisson_nll"]),
            "mean_poisson_deviance": float(main_test_metrics["mean_poisson_deviance"]),
            "mae": float(main_test_metrics["mae"]),
            "rmse": float(main_test_metrics["rmse"]),
            "aggregate_bias": float(main_test_metrics["aggregate_bias"]),
            "relative_aggregate_bias": float(main_test_metrics["relative_aggregate_bias"]),
            "mean_target": float(main_test_metrics["mean_target"]),
            "mean_prediction": float(main_test_metrics["mean_prediction"]),
        },
        "train_metrics_pooled_hawkes": {
            "poisson_loglik": float(main_train_metrics["poisson_loglik"]),
            "mean_poisson_nll": float(main_train_metrics["mean_poisson_nll"]),
            "mean_poisson_deviance": float(main_train_metrics["mean_poisson_deviance"]),
            "mae": float(main_train_metrics["mae"]),
            "rmse": float(main_train_metrics["rmse"]),
            "aggregate_bias": float(main_train_metrics["aggregate_bias"]),
        },
        "fit_params": {
            "alpha_l2": ALPHA_L2,
            "scale_l2": SCALE_L2,
            "c": float(main_c),
            "alpha_norm": float(np.linalg.norm(main_alpha)),
            "alpha": main_alpha.tolist(),
            "feature_names": list(HAWKES_FEATURES),
            "half_lives": list(MAIN_HALF_LIVES),
            "n_params": 1 + len(HAWKES_FEATURES) * len(MAIN_HALF_LIVES),
        },
        "user_level_vs_personalized_gp": {
            "share_pos_overall": float((merged["delta_ll"] > 0).mean()),
            "mean_delta_ll": float(merged["delta_ll"].mean()),
            "median_delta_ll": float(merged["delta_ll"].median()),
            "by_bucket": bucket_summary.reset_index().to_dict(orient="records"),
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Pooled Hawkes (main, half_lives = {MAIN_HALF_LIVES}) ===")
    print(f"  c = {main_c:.4f}, ||α|| = {np.linalg.norm(main_alpha):.4f}")
    print(f"  test NLL = {main_test_metrics['mean_poisson_nll']:.5f}")
    print(f"  vs Personalized GP: Δ LL = {main_test_metrics['poisson_loglik'] - pers_metrics['poisson_loglik']:+.0f} нат")
    print(f"  share(pooled > pers) = {summary['user_level_vs_personalized_gp']['share_pos_overall']:.4f}")
    print(f"  mean Δ LL per user   = {summary['user_level_vs_personalized_gp']['mean_delta_ll']:+.4f}")
    print(f"\nArtifacts saved to {OUT_DIR}/")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

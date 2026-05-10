"""Train-length scan engine for chapter 11.

For each n ∈ N_GRID, sample m random consecutive intervals of n days from
[GLOBAL_START, GLOBAL_END]. Within each interval: first 2n/3 days are train,
last n/3 are test. All probabilistic models are fitted on train and evaluated
on test. Per-run test NLL is persisted to CSV.

The GBDT model is included as well: feature engineering is performed once
on the full analysis window, and per-interval cost is reduced to a slice
plus a HistGradientBoosting fit. All n values are multiples of 3.
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

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

from src.diploma_baselines.data import filter_date_range, load_daily_grid
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
    fit_joint_hawkes,
    fit_pooled_additive_multi_kernel_hawkes,
    fit_pooled_hawkes,
    predict_pooled_additive_multi_kernel_hawkes,
)
from src.diploma_experimental.gbdt import SOURCE_FEATURES, build_feature_panel


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

GLOBAL_START = pd.Timestamp("2025-01-15")
GLOBAL_END = pd.Timestamp("2025-10-31")
BASE_SEED = 42

# 10 values, multiples of 3, log-spaced from 15 to 198
N_GRID = [15, 21, 27, 36, 48, 63, 84, 114, 153, 198]
# m schedule: ~50 at n=14, ~10 at n=100, gentle decay
M_GRID = [50, 35, 30, 25, 20, 18, 14, 10, 8, 6]
assert len(N_GRID) == len(M_GRID)

OUT_DIR = Path("diploma/reports/11_train_length_scan")

MODEL_LABELS = [
    "Global Poisson",
    "Rolling Poisson",
    "Rolling Seasonal",
    "Personalized Gamma-Poisson",
    "Scaled-baseline Hawkes",
    "Joint Hawkes (λ_u + α)",
    "Pooled Hawkes (c·b_t + α^T s)",
    "GBDT (experimental)",
]


# ----------------------------------------------------------------------
# Per-interval evaluation
# ----------------------------------------------------------------------
def evaluate_interval(
    full_df: pd.DataFrame,
    daily_mean_full: pd.Series,
    interval_start: pd.Timestamp,
    n_days: int,
    user_states_cache: UserStatesCache,
    gbdt_x_full: np.ndarray | None = None,
    gbdt_dates_full: np.ndarray | None = None,
    gbdt_targets_full: np.ndarray | None = None,
):
    """Fit and evaluate all models on one (start, n_days) interval. Returns dict."""
    train_len = (n_days // 3) * 2
    train_end = interval_start + pd.Timedelta(days=train_len - 1)
    test_start = train_end + pd.Timedelta(days=1)
    test_end = interval_start + pd.Timedelta(days=n_days - 1)

    block_df = full_df.loc[
        (full_df["event_date"] >= interval_start) & (full_df["event_date"] <= test_end)
    ]
    train_df = block_df[block_df["event_date"] <= train_end].copy()
    test_df = block_df[block_df["event_date"] >= test_start].copy()

    out = {
        "interval_start": str(interval_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "test_rows": int(len(test_df)),
    }

    if len(test_df) == 0 or len(train_df) == 0:
        for label in MODEL_LABELS:
            out[label] = float("nan")
        return out

    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    # 1. Global Poisson
    gp = GlobalPoissonModel().fit(y_train)
    pred = gp.predict(len(test_df))
    out["Global Poisson"] = float(evaluate_count_forecast(y_test, pred)["mean_poisson_nll"])

    # 2. Rolling Poisson
    rp = GlobalRollingMeanPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(daily_mean_full)
    pred = rp.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)
    out["Rolling Poisson"] = float(evaluate_count_forecast(y_test, pred)["mean_poisson_nll"])

    # 3. Rolling Seasonal
    train_daily_mean = train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean, daily_mean_full
    )
    base_train = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    base_test = rs.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)
    out["Rolling Seasonal"] = float(evaluate_count_forecast(y_test, base_test)["mean_poisson_nll"])

    # 4. Personalized GP
    scaler = PersonalizedGammaPoissonScaler().fit(
        train_df["user_id"].to_numpy(), y_train, base_train
    )
    train_pers = scaler.predict(train_df["user_id"].to_numpy(), base_train, method="posterior_mean")
    test_pers = scaler.predict(test_df["user_id"].to_numpy(), base_test, method="posterior_mean")
    out["Personalized Gamma-Poisson"] = float(evaluate_count_forecast(y_test, test_pers)["mean_poisson_nll"])

    # Build Hawkes states for train and test from cache
    X_train = user_states_cache.gather_for(train_df)
    X_test = user_states_cache.gather_for(test_df)
    train_groups = train_df.groupby("user_id", sort=False).indices
    test_groups = test_df.groupby("user_id", sort=False).indices

    # 5. Scaled-baseline Hawkes (staged: base = train_pers, c + alpha)
    train_state_blocks: list = []
    train_y_blocks: list = []
    train_base_blocks: list = []
    test_state_blocks: list = []
    test_y_blocks: list = []
    test_base_blocks: list = []
    train_pred_by_user = {int(uid): train_pers[idx] for uid, idx in train_groups.items()}
    test_pred_by_user = {int(uid): test_pers[idx] for uid, idx in test_groups.items()}

    for uid, idx in train_groups.items():
        train_state_blocks.append(X_train[idx])
        train_y_blocks.append(y_train[idx])
        train_base_blocks.append(train_pred_by_user[int(uid)])
    for uid, idx in test_groups.items():
        test_state_blocks.append(X_test[idx])
        test_y_blocks.append(y_test[idx])
        test_base_blocks.append(test_pred_by_user[int(uid)])

    try:
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
            max_iter=200,
        )
        test_preds = []
        for states, base in zip(test_state_blocks, test_base_blocks):
            lam, _ = predict_pooled_additive_multi_kernel_hawkes(
                hawkes, states=states, base_lambda=base
            )
            test_preds.append(lam)
        out["Scaled-baseline Hawkes"] = float(
            evaluate_count_forecast(y_test, np.concatenate(test_preds))["mean_poisson_nll"]
        )
        # mean per-user multiplier m_u = c * mu_u^EB averaged over train users
        train_mu_eb_per_user = scaler.user_stats_["mu_posterior_mean"].to_numpy(dtype=float)
        m_u_staged_mean = float(hawkes.base_scale * train_mu_eb_per_user.mean())
        out["Scaled-baseline Hawkes_alpha_norm"] = float(np.linalg.norm(hawkes.alpha))
        out["Scaled-baseline Hawkes_c"] = float(hawkes.base_scale)
        out["Scaled-baseline Hawkes_m_u_mean"] = m_u_staged_mean
    except Exception as exc:
        out["Scaled-baseline Hawkes"] = float("nan")
        out["Scaled-baseline Hawkes_alpha_norm"] = float("nan")
        out["Scaled-baseline Hawkes_c"] = float("nan")
        out["Scaled-baseline Hawkes_m_u_mean"] = float("nan")

    # 6. Joint Hawkes (λ_u + α)
    train_uids = train_df["user_id"].to_numpy()
    unique_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
    n_users = int(len(unique_uids))
    uid_to_idx = {int(u): i for i, u in enumerate(unique_uids)}
    test_user_idx = np.array(
        [uid_to_idx.get(int(u), -1) for u in test_df["user_id"].to_numpy()], dtype=int
    )
    try:
        joint_res = fit_joint_hawkes(
            user_idx=train_user_idx,
            y=y_train,
            b=base_train,
            states=X_train.astype(float),
            n_users=n_users,
            lambda_l2=1.0,
            alpha_l2=1e-4,
            max_iter=200,
        )
        lam_u, alpha = joint_res.lam_u, joint_res.alpha
        lam_u_for_test = np.where(test_user_idx >= 0, lam_u[np.maximum(test_user_idx, 0)], 1.0)
        lam_test = lam_u_for_test * base_test + X_test.astype(float) @ alpha
        out["Joint Hawkes (λ_u + α)"] = float(
            evaluate_count_forecast(y_test, np.clip(lam_test, 1e-8, None))["mean_poisson_nll"]
        )
        out["Joint Hawkes (λ_u + α)_alpha_norm"] = float(np.linalg.norm(alpha))
        out["Joint Hawkes (λ_u + α)_m_u_mean"] = float(lam_u.mean())
        out["Joint Hawkes (λ_u + α)_lambda_median"] = float(np.median(lam_u))
    except Exception:
        out["Joint Hawkes (λ_u + α)"] = float("nan")
        out["Joint Hawkes (λ_u + α)_alpha_norm"] = float("nan")
        out["Joint Hawkes (λ_u + α)_m_u_mean"] = float("nan")
        out["Joint Hawkes (λ_u + α)_lambda_median"] = float("nan")

    # 7. Pooled Hawkes (c · b_t + α^T s)
    try:
        pooled_res = fit_pooled_hawkes(
            y=y_train, b=base_train, states=X_train.astype(float),
            alpha_l2=1e-4, scale_l2=10.0, max_iter=300,
        )
        c_fit, alpha_fit = pooled_res.c, pooled_res.alpha
        lam_test = c_fit * base_test + X_test.astype(float) @ alpha_fit
        out["Pooled Hawkes (c·b_t + α^T s)"] = float(
            evaluate_count_forecast(y_test, np.clip(lam_test, 1e-8, None))["mean_poisson_nll"]
        )
        # m_u for Pooled Hawkes is just c (single global multiplier, no per-user)
        out["Pooled Hawkes (c·b_t + α^T s)_alpha_norm"] = float(np.linalg.norm(alpha_fit))
        out["Pooled Hawkes (c·b_t + α^T s)_c"] = float(c_fit)
        out["Pooled Hawkes (c·b_t + α^T s)_m_u_mean"] = float(c_fit)
    except Exception:
        out["Pooled Hawkes (c·b_t + α^T s)"] = float("nan")
        out["Pooled Hawkes (c·b_t + α^T s)_alpha_norm"] = float("nan")
        out["Pooled Hawkes (c·b_t + α^T s)_c"] = float("nan")
        out["Pooled Hawkes (c·b_t + α^T s)_m_u_mean"] = float("nan")

    # 8. GBDT (experimental): slice the precomputed feature panel by date,
    #    fit HistGradientBoosting on the train slice, evaluate on test slice.
    if gbdt_x_full is not None and gbdt_dates_full is not None and gbdt_targets_full is not None:
        try:
            mask_tr = (gbdt_dates_full >= np.datetime64(interval_start)) & (
                gbdt_dates_full <= np.datetime64(train_end)
            )
            mask_te = (gbdt_dates_full >= np.datetime64(test_start)) & (
                gbdt_dates_full <= np.datetime64(test_end)
            )
            x_tr = gbdt_x_full[mask_tr]
            y_tr = gbdt_targets_full[mask_tr]
            x_te = gbdt_x_full[mask_te]
            y_te = gbdt_targets_full[mask_te]
            model = HistGradientBoostingRegressor(
                loss="poisson",
                max_depth=5,
                learning_rate=0.05,
                max_iter=200,
                min_samples_leaf=40,
                random_state=42,
            )
            model.fit(x_tr, y_tr)
            pred = np.clip(model.predict(x_te), 1e-8, None)
            out["GBDT (experimental)"] = float(
                evaluate_count_forecast(y_te, pred)["mean_poisson_nll"]
            )
        except Exception:
            out["GBDT (experimental)"] = float("nan")
    else:
        out["GBDT (experimental)"] = float("nan")

    return out


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def sample_starts(n_days: int, m: int, seed: int):
    earliest = GLOBAL_START
    latest = GLOBAL_END - pd.Timedelta(days=n_days - 1)
    n_possible = (latest - earliest).days + 1
    rng = np.random.default_rng(seed)
    if n_possible <= m:
        offsets = list(range(n_possible))
    else:
        offsets = sorted(rng.choice(n_possible, size=m, replace=False).tolist())
    return [earliest + pd.Timedelta(days=int(o)) for o in offsets]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES, *SOURCE_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    # Build GBDT feature panel once for the full analysis window — slicing
    # per-interval is then ~free, and only the GBDT fit itself runs per run.
    print("Building GBDT feature panel for full analysis window...")
    t_panel0 = time.time()
    gbdt_x_full, gbdt_index_full, _ = build_feature_panel(
        full_df=full_df,
        analysis_start=GLOBAL_START,
        analysis_end=GLOBAL_END,
        target_col=TARGET_COL,
        source_features=SOURCE_FEATURES,
    )
    gbdt_dates_full = gbdt_index_full["event_date"].to_numpy(dtype="datetime64[ns]")
    gbdt_targets_full = gbdt_index_full["target"].to_numpy(dtype=float)
    print(
        f"  built panel: {gbdt_x_full.shape[0]:,} rows × {gbdt_x_full.shape[1]} feats "
        f"in {time.time() - t_panel0:.1f}s"
    )

    # Build Hawkes states cache once (per-user, exp-decay)
    print("Building Hawkes states cache...")
    user_states_cache = build_user_states_cache(
        full_df, features=HAWKES_FEATURES, half_lives=HALF_LIVES,
    )
    print(f"  cached states for {len(user_states_cache.states_per_user):,} users")

    # === Run scan ===
    rows: list[dict] = []
    total_runs = sum(M_GRID)
    run_idx = 0
    t_start = time.time()

    for n_days, m in zip(N_GRID, M_GRID):
        starts = sample_starts(n_days, m, seed=BASE_SEED + n_days)
        actual_m = len(starts)
        print(f"\n=== n={n_days}d  ({actual_m} intervals) ===")
        for k, start in enumerate(starts):
            run_idx += 1
            t0 = time.time()
            row = evaluate_interval(
                full_df=full_df,
                daily_mean_full=daily_mean_full,
                interval_start=start,
                n_days=n_days,
                user_states_cache=user_states_cache,
                gbdt_x_full=gbdt_x_full,
                gbdt_dates_full=gbdt_dates_full,
                gbdt_targets_full=gbdt_targets_full,
            )
            row["n_days"] = int(n_days)
            row["m_idx"] = int(k)
            rows.append(row)
            elapsed = time.time() - t0
            eta_total = (time.time() - t_start) / run_idx * (total_runs - run_idx)
            print(
                f"  [{run_idx:>3}/{total_runs}] n={n_days:>3} k={k:>2} "
                f"start={row['interval_start']} test_rows={row['test_rows']:>6,} "
                f"NLL: GP={row['Personalized Gamma-Poisson']:.4f} Joint={row['Joint Hawkes (λ_u + α)']:.4f} "
                f"Pooled={row['Pooled Hawkes (c·b_t + α^T s)']:.4f} "
                f"GBDT={row['GBDT (experimental)']:.4f}  "
                f"({elapsed:.1f}s, ETA {eta_total/60:.1f}m)"
            )
            # Save incrementally every few runs
            if run_idx % 20 == 0:
                pd.DataFrame(rows).to_csv(OUT_DIR / "scan_results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "scan_results.csv", index=False)

    # Save metadata
    metadata = {
        "n_grid": N_GRID,
        "m_grid": M_GRID,
        "model_labels": MODEL_LABELS,
        "global_start": str(GLOBAL_START.date()),
        "global_end": str(GLOBAL_END.date()),
        "base_seed": BASE_SEED,
        "total_runs": int(len(df)),
        "elapsed_seconds": float(time.time() - t_start),
    }
    with open(OUT_DIR / "scan_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nDone in {time.time() - t_start:.1f}s. Saved {len(df)} rows to {OUT_DIR / 'scan_results.csv'}")


if __name__ == "__main__":
    main()

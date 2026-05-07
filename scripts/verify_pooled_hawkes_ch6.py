"""Independent re-verification of Pooled Hawkes on ch.6 protocol.

Loads the saved (c, alpha) from summary.json and re-computes test predictions
from scratch, then checks NLL matches the saved value.

Also recomputes:
  - Rolling Seasonal NLL (alpha=0, c=1, b_t only)        -> sanity vs chapter 8
  - Optimal-c NLL (alpha=0, c free) on raw b_t           -> compare with c=1
  - Personalized GP-like baseline (mu_eb * b_t)          -> sanity vs chapter 8
  - Pooled Hawkes from saved params                      -> verify 0.4025
  - Pooled Hawkes refit                                  -> verify reproducibility
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
mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.diploma_baselines.data import filter_date_range, load_daily_grid
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_basis_states,
)


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

TRAIN_START = pd.Timestamp("2025-01-15")
TRAIN_END = pd.Timestamp("2025-08-09")
TEST_START = pd.Timestamp("2025-08-10")
TEST_END = pd.Timestamp("2025-09-30")


def main():
    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    train_df = filter_date_range(full_df, start_date=TRAIN_START, end_date=TRAIN_END).copy()
    test_df = filter_date_range(full_df, start_date=TEST_START, end_date=TEST_END).copy()
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

    print(f"\n  mean(y_train)={y_train.mean():.4f}, mean(y_test)={y_test.mean():.4f}")
    print(f"  mean(b_train)={b_train.mean():.4f}, mean(b_test)={b_test.mean():.4f}")

    # ---- 1. Rolling Seasonal (b_t only, c=1) ----
    nll_rs = evaluate_count_forecast(y_test, np.clip(b_test, 1e-8, None))["mean_poisson_nll"]
    print(f"\n[1] Rolling Seasonal (c=1, alpha=0):     NLL = {nll_rs:.4f}  (expected ~0.4576 per ch.8)")

    # ---- 2. Optimal-c only (alpha=0, c free) ----
    from scipy.optimize import minimize_scalar
    def loss_c(c):
        lam = np.clip(float(c) * b_train, 1e-8, None)
        return float(np.sum(lam - y_train * np.log(lam)))
    res_c = minimize_scalar(loss_c, bounds=(0.001, 10.0), method="bounded")
    c_opt = float(res_c.x)
    nll_c = evaluate_count_forecast(y_test, np.clip(c_opt * b_test, 1e-8, None))["mean_poisson_nll"]
    print(f"[2] Optimal-c on raw b_t (alpha=0):       c={c_opt:.4f}  NLL = {nll_c:.4f}  (sanity)")

    # ---- 3. Personalized GP independently fit ----
    scaler = PersonalizedGammaPoissonScaler().fit(
        train_df["user_id"].to_numpy(),
        y_train, b_train,
    )
    pers_test_pred = scaler.predict(test_df["user_id"].to_numpy(), b_test, method="posterior_mean")
    nll_pers = evaluate_count_forecast(y_test, np.clip(pers_test_pred, 1e-8, None))["mean_poisson_nll"]
    print(f"[3] Personalized GP (mu_u_EB * b_t):      NLL = {nll_pers:.4f}  (expected ~0.4096 per ch.8)")
    print(f"    EB params: alpha0={scaler.alpha_:.4f}, beta0={scaler.beta_:.4f}, prior_mean={scaler.prior_mean_:.4f}")

    # ---- 4. Build Hawkes states. Carefully causal. ----
    print("\nBuilding Hawkes states...")
    beta = np.log(2.0) / np.asarray(HALF_LIVES, dtype=float)
    n_alpha = len(HAWKES_FEATURES) * len(HALF_LIVES)

    train_dates_full = train_df["event_date"].to_numpy(dtype="datetime64[ns]")
    test_dates_full = test_df["event_date"].to_numpy(dtype="datetime64[ns]")

    X_train = np.zeros((len(train_df), n_alpha), dtype=np.float64)
    X_test = np.zeros((len(test_df), n_alpha), dtype=np.float64)

    train_groups = train_df.groupby("user_id", sort=False).indices
    test_groups = test_df.groupby("user_id", sort=False).indices

    # Verify causal: for each user, recompute states day-by-day
    n_state_checked = 0
    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float64)
        full_dates = full_user["event_date"].to_numpy(dtype="datetime64[ns]")
        full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}

        # Causality spot-check: for first user only, verify states[10] matches manual recompute
        if n_state_checked == 0 and len(full_user) >= 10:
            decay = np.exp(-beta)
            manual = np.zeros((5, 2), dtype=float)
            for t_check in range(1, 10):
                manual = manual * decay + x_full[t_check - 1].reshape(5, 1)
            assert np.allclose(states_full[10][:].reshape(5, 2), manual.reshape(5, 2) * decay + x_full[9].reshape(5, 1)), "build_basis_states bug"
            print(f"  causality spot-check OK on user {int(user_id)}")
            n_state_checked = 1

        if int(user_id) in train_groups:
            idx_in_block = train_groups[int(user_id)]
            wanted = train_dates_full[idx_in_block]
            rows_in_full = np.array([full_to_idx[pd.Timestamp(d).normalize()] for d in wanted], dtype=int)
            X_train[idx_in_block] = states_full[rows_in_full]
        if int(user_id) in test_groups:
            idx_in_block = test_groups[int(user_id)]
            wanted = test_dates_full[idx_in_block]
            rows_in_full = np.array([full_to_idx[pd.Timestamp(d).normalize()] for d in wanted], dtype=int)
            X_test[idx_in_block] = states_full[rows_in_full]

    print(f"  X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    print(f"  mean(||s||) on train rows = {np.linalg.norm(X_train, axis=1).mean():.4f}")
    print(f"  mean(||s||) on test rows  = {np.linalg.norm(X_test, axis=1).mean():.4f}")

    # ---- 5. Pooled Hawkes from SAVED params ----
    saved = json.loads(Path("diploma/reports/pooled_hawkes_ch6/summary.json").read_text())
    c_saved = saved["fit_params"]["c"]
    alpha_saved = np.asarray(saved["fit_params"]["alpha"], dtype=float)
    print(f"\n[4] Pooled Hawkes from SAVED (c={c_saved:.4f}, ||a||={np.linalg.norm(alpha_saved):.4f})")
    lam_test_saved = np.clip(c_saved * b_test + X_test @ alpha_saved, 1e-8, None)
    nll_saved = evaluate_count_forecast(y_test, lam_test_saved)["mean_poisson_nll"]
    print(f"    NLL recomputed = {nll_saved:.4f}  (saved said {saved['test_metrics_pooled_hawkes']['mean_poisson_nll']:.4f})")

    # ---- 6. Pooled Hawkes refit from scratch ----
    def fg(p):
        c = float(p[0]); alpha = np.asarray(p[1:], dtype=float)
        lam = np.clip(c * b_train + X_train @ alpha, 1e-8, None)
        nll = float(np.sum(lam - y_train * np.log(lam))
                    + 1e-4 * np.sum(alpha**2)
                    + 10.0 * (c - 1.0) ** 2)
        d = 1.0 - y_train / lam
        a_grad = X_train.T @ d + 2.0 * 1e-4 * alpha
        c_grad = float(np.sum(b_train * d) + 2.0 * 10.0 * (c - 1.0))
        return nll, np.concatenate([[c_grad], a_grad])

    init = np.concatenate([[1.0], np.full(n_alpha, 0.01)])
    bounds = [(0.001, 50.0)] + [(0.0, 10.0)] * n_alpha
    print("\n[5] Refitting Pooled Hawkes from scratch...")
    res = minimize(lambda p: fg(p)[0], init, method="L-BFGS-B",
                   jac=lambda p: fg(p)[1], bounds=bounds, options={"maxiter": 500})
    c_refit = float(res.x[0]); alpha_refit = np.asarray(res.x[1:], dtype=float)
    print(f"    c_refit={c_refit:.4f}, ||alpha||={np.linalg.norm(alpha_refit):.4f}")
    print(f"    max |alpha_refit - alpha_saved| = {np.max(np.abs(alpha_refit - alpha_saved)):.6f}")
    print(f"    |c_refit - c_saved| = {abs(c_refit - c_saved):.6f}")
    lam_test_refit = np.clip(c_refit * b_test + X_test @ alpha_refit, 1e-8, None)
    nll_refit = evaluate_count_forecast(y_test, lam_test_refit)["mean_poisson_nll"]
    print(f"    NLL_refit = {nll_refit:.4f}")

    # ---- 7. Decomposition: how much does each part contribute? ----
    print("\n[6] Decomposition of Pooled Hawkes prediction on test:")
    base_part = c_saved * b_test
    hawkes_part = X_test @ alpha_saved
    print(f"    mean(c*b_t)        = {base_part.mean():.4f}")
    print(f"    mean(alpha^T s)    = {hawkes_part.mean():.4f}")
    print(f"    mean(total)        = {(base_part + hawkes_part).mean():.4f}")
    print(f"    mean(y_test)       = {y_test.mean():.4f}")

    # ---- 8. Counterfactual: what if we drop Hawkes (alpha=0) and refit only c on train? ----
    nll_no_hawkes = evaluate_count_forecast(y_test, np.clip(c_opt * b_test, 1e-8, None))["mean_poisson_nll"]
    print(f"\n[7] Counterfactual NLL with alpha=0 (just c*b_t)    = {nll_no_hawkes:.4f}")
    print(f"    NLL with full Pooled Hawkes (c={c_saved:.4f}, alpha) = {nll_saved:.4f}")
    print(f"    Hawkes contribution: {nll_no_hawkes - nll_saved:+.4f} nat/n")

    # ---- 9. Sanity: did Pooled Hawkes use any feature that might leak? ----
    print(f"\n[8] Feature decomposition of contribution to test mean:")
    print(f"    {'feature':<15s} {'hl=1':>10s} {'hl=3':>10s}")
    for f_idx, fname in enumerate(HAWKES_FEATURES):
        a1 = alpha_saved[f_idx * 2 + 0]
        a3 = alpha_saved[f_idx * 2 + 1]
        contrib1 = (X_test[:, f_idx * 2 + 0] * a1).mean()
        contrib3 = (X_test[:, f_idx * 2 + 1] * a3).mean()
        print(f"    {fname:<15s} a={a1:.4f} m={contrib1:.4f}    a={a3:.4f} m={contrib3:.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

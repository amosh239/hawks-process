"""Diagnose why staged scaled-baseline Hawkes degenerates on 14d blocks.

Key distinction (this is what we got wrong before):
  * STAGED Hawkes (pipeline) uses base_t = mu_u^EB * b_t (already personalized).
  * JOINT Hawkes (ch.14) uses base_t = b_t (raw rolling-seasonal), and fits
    a free per-user lambda_u + alpha jointly.

We explicitly distinguish these two bases below.
"""

from __future__ import annotations

import os
import sys
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
HAWKES_HALF_LIVES = (1.0, 3.0)
HAWKES_FEATURES = tuple(FEATURE_NAMES)


def assemble_block(full_df, block_start, train_len=14, test_len=7):
    block_end = block_start + pd.Timedelta(days=train_len + test_len - 1)
    train_end = block_start + pd.Timedelta(days=train_len - 1)
    test_start = train_end + pd.Timedelta(days=1)
    block_df = filter_date_range(full_df, start_date=block_start, end_date=block_end)
    train_df = block_df[block_df["event_date"] <= train_end].copy()
    test_df = block_df[block_df["event_date"] >= test_start].copy()
    return train_df, test_df, block_start, block_end, train_end


def build_per_row_arrays(full_df, train_df, test_df, block_start, block_end, train_end,
                        rs_model, scaler):
    """Build, for both train and test, per-row arrays:
       - X (Hawkes states)
       - y
       - b_raw (rolling-seasonal baseline, no user scaling)
       - mu_eb (EB posterior mean per user, broadcast to row)
       - user_idx (0..n_train_users-1, with -1 for test users absent in train)
    """
    beta = np.log(2.0) / np.asarray(HAWKES_HALF_LIVES, dtype=float)
    full_groups = full_df.groupby("user_id", sort=False)

    block_start64 = np.datetime64(block_start)
    train_end64 = np.datetime64(train_end)
    block_end64 = np.datetime64(block_end)

    train_X_rows, train_y_rows, train_b_rows, train_mu_rows, train_uid_rows = [], [], [], [], []
    test_X_rows, test_y_rows, test_b_rows, test_mu_rows, test_uid_rows = [], [], [], [], []

    mu_series = scaler.user_stats_["mu_posterior_mean"]
    mu_eb_by_uid = {int(uid): float(val) for uid, val in mu_series.items()}
    prior_mean = float(scaler.prior_mean_)

    rs_dates_train_cache = {}
    rs_dates_test_cache = {}

    for user_id, full_user in full_groups:
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        full_dates = full_user["event_date"].to_numpy(dtype="datetime64[ns]")

        train_mask = (full_dates >= block_start64) & (full_dates <= train_end64)
        test_mask = (full_dates > train_end64) & (full_dates <= block_end64)

        if train_mask.any():
            states_train = states_full[train_mask]
            y_user = full_user[TARGET_COL].to_numpy(dtype=float)[train_mask]
            dates_user = full_dates[train_mask]
            b_raw = rs_model.predict_for_dates(pd.Series(dates_user)).to_numpy(dtype=float)
            mu_eb = mu_eb_by_uid.get(int(user_id), prior_mean)
            train_X_rows.append(states_train)
            train_y_rows.append(y_user)
            train_b_rows.append(b_raw)
            train_mu_rows.append(np.full(len(y_user), mu_eb))
            train_uid_rows.append(np.full(len(y_user), int(user_id)))

        if test_mask.any():
            states_test = states_full[test_mask]
            y_user = full_user[TARGET_COL].to_numpy(dtype=float)[test_mask]
            dates_user = full_dates[test_mask]
            b_raw = rs_model.predict_for_dates(pd.Series(dates_user)).to_numpy(dtype=float)
            mu_eb = mu_eb_by_uid.get(int(user_id), prior_mean)
            test_X_rows.append(states_test)
            test_y_rows.append(y_user)
            test_b_rows.append(b_raw)
            test_mu_rows.append(np.full(len(y_user), mu_eb))
            test_uid_rows.append(np.full(len(y_user), int(user_id)))

    Xtr = np.vstack(train_X_rows)
    ytr = np.concatenate(train_y_rows)
    btr = np.concatenate(train_b_rows)
    mutr = np.concatenate(train_mu_rows)
    uidtr = np.concatenate(train_uid_rows)

    Xte = np.vstack(test_X_rows)
    yte = np.concatenate(test_y_rows)
    bte = np.concatenate(test_b_rows)
    mute = np.concatenate(test_mu_rows)
    uidte = np.concatenate(test_uid_rows)

    return (Xtr, ytr, btr, mutr, uidtr), (Xte, yte, bte, mute, uidte)


def staged_loss_grad(c, alpha, X, y, base, alpha_l2, scale_l2):
    lam = np.clip(c * base + X @ alpha, 1e-8, None)
    nll = float(np.sum(lam - y * np.log(lam))
                + alpha_l2 * np.sum(alpha**2)
                + scale_l2 * (c - 1.0) ** 2)
    alpha_grad = X.T @ (1.0 - y / lam) + 2.0 * alpha_l2 * alpha
    c_grad = float(np.sum(base * (1.0 - y / lam)) + 2.0 * scale_l2 * (c - 1.0))
    return nll, c_grad, alpha_grad


def fit_staged(X, y, base, alpha_init, scale_init, alpha_l2=1e-4, scale_l2=10.0,
               max_iter=300, alpha_lower=0.0):
    n_alpha = X.shape[1]

    def fg(p):
        c = float(p[0]); a = np.asarray(p[1:], dtype=float)
        nll, cg, ag = staged_loss_grad(c, a, X, y, base, alpha_l2, scale_l2)
        return nll, np.concatenate([[cg], ag])

    init = np.concatenate([[float(scale_init)], np.asarray(alpha_init, dtype=float).reshape(-1)])
    bounds = [(0.1, 5.0)] + [(float(alpha_lower), 10.0)] * n_alpha
    res = minimize(lambda p: fg(p)[0], init, method="L-BFGS-B",
                   jac=lambda p: fg(p)[1], bounds=bounds, options={"maxiter": max_iter})
    c = float(res.x[0]); alpha = np.asarray(res.x[1:], dtype=float)
    nll, cg, ag = staged_loss_grad(c, alpha, X, y, base, alpha_l2, scale_l2)
    return c, alpha, nll, cg, ag, bool(res.success), int(res.nit)


def fit_joint_lambda_alpha(X, y, b_raw, user_idx, n_users, gamma=1.0, alpha_l2=1e-4):
    n_alpha = X.shape[1]

    def fg(p):
        lam_u = p[:n_users]; alpha = p[n_users:]
        per_row = lam_u[user_idx]
        lam = np.clip(per_row * b_raw + X @ alpha, 1e-8, None)
        nll = float(np.sum(lam - y * np.log(lam))
                    + gamma * np.sum((lam_u - 1.0) ** 2)
                    + alpha_l2 * np.sum(alpha ** 2))
        d = 1.0 - y / lam
        a_grad = X.T @ d + 2.0 * alpha_l2 * alpha
        l_grad = np.bincount(user_idx, weights=b_raw * d, minlength=n_users) + 2.0 * gamma * (lam_u - 1.0)
        return nll, np.concatenate([l_grad, a_grad])

    init = np.concatenate([np.ones(n_users), np.full(n_alpha, 0.01)])
    bounds = [(0.001, 50.0)] * n_users + [(0.0, 10.0)] * n_alpha
    res = minimize(lambda p: fg(p)[0], init, method="L-BFGS-B",
                   jac=lambda p: fg(p)[1], bounds=bounds, options={"maxiter": 500})
    lam_u = res.x[:n_users]; alpha = res.x[n_users:]
    return lam_u, alpha, float(res.fun), bool(res.success)


def main():
    data_path = "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv"
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    print("Loading data...")
    full_df = load_daily_grid(data_path, value_cols=cols)
    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    block_start = pd.Timestamp("2025-04-30")
    train_df, test_df, bstart, bend, tend = assemble_block(full_df, block_start)
    print(f"\nBlock: {bstart.date()} .. {bend.date()} (train end {tend.date()}, test rows {len(test_df):,})")

    train_daily_mean = train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    rs = GlobalRollingSeasonalPoissonModel(window_size=7, min_periods=1).fit(train_daily_mean, daily_mean_full)
    base_train_raw = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    scaler = PersonalizedGammaPoissonScaler().fit(
        train_df["user_id"].to_numpy(),
        train_df[TARGET_COL].to_numpy(),
        base_train_raw,
    )
    print(f"  EB (alpha0, beta0) = ({scaler.alpha_:.4f}, {scaler.beta_:.4f})")

    (Xtr, ytr, btr, mutr, uidtr), (Xte, yte, bte, mute, uidte) = build_per_row_arrays(
        full_df, train_df, test_df, bstart, bend, tend, rs, scaler)

    print(f"  train rows={len(ytr):,}, test rows={len(yte):,}")
    print(f"  raw b_t: mean={btr.mean():.4f}, range=[{btr.min():.4f}, {btr.max():.4f}]")
    print(f"  mu_eb:   mean={mutr.mean():.4f}, q05={np.quantile(mutr, 0.05):.4f}, q95={np.quantile(mutr, 0.95):.4f}")

    base_eb_train = mutr * btr  # this is what staged sees
    base_eb_test = mute * bte
    print(f"  EB-scaled base on train: mean={base_eb_train.mean():.4f}")

    # ----------------------------------------------------------------
    # 1. Staged Hawkes — current pipeline. base = mu_eb * b_t.
    # ----------------------------------------------------------------
    print("\n=== [STAGED] base = mu_eb * b_t (current pipeline) ===")
    c, alpha, nll, cg, ag, ok, nit = fit_staged(
        Xtr, ytr, base_eb_train, alpha_init=np.full(10, 0.01), scale_init=1.0)
    print(f"  c={c:.6f}, ||alpha||={np.linalg.norm(alpha):.6f}")
    print(f"  c-grad = {cg:.4e}, alpha-grad: pos={int((ag>0).sum())}/10")
    print(f"  alpha-grads: {np.round(ag, 1)}")
    test_lam = c * base_eb_test + Xte @ alpha
    nll_te = evaluate_count_forecast(yte, np.clip(test_lam, 1e-8, None))["mean_poisson_nll"]
    print(f"  TEST NLL = {nll_te:.4f}")

    # ----------------------------------------------------------------
    # 2. Joint Hawkes (ch.14): base = b_t raw, free lambda_u + alpha.
    # ----------------------------------------------------------------
    print("\n=== [JOINT] base = b_t raw, free lambda_u + alpha (gamma=1) ===")
    unique_uids, train_user_idx = np.unique(uidtr, return_inverse=True)
    n_users = len(unique_uids)
    lam_u, alpha_j, train_loss_j, ok_j = fit_joint_lambda_alpha(
        Xtr, ytr, btr, train_user_idx, n_users, gamma=1.0)
    print(f"  mean(lambda_u)={lam_u.mean():.4f}, std={lam_u.std():.4f}")
    print(f"  ||alpha||={np.linalg.norm(alpha_j):.4f}, alpha={np.round(alpha_j, 4)}")
    # Predict on test
    uid_to_idx = {int(u): i for i, u in enumerate(unique_uids)}
    test_user_idx = np.array([uid_to_idx.get(int(u), -1) for u in uidte])
    lam_u_for_test = np.where(test_user_idx >= 0, lam_u[np.maximum(test_user_idx, 0)], 1.0)
    test_lam_j = lam_u_for_test * bte + Xte @ alpha_j
    nll_te_j = evaluate_count_forecast(yte, np.clip(test_lam_j, 1e-8, None))["mean_poisson_nll"]
    print(f"  TEST NLL = {nll_te_j:.4f}")

    # ----------------------------------------------------------------
    # 3. CRUCIAL: staged with raw b_t base (no EB pre-shrinkage).
    # ----------------------------------------------------------------
    print("\n=== [STAGED-on-raw] base = b_t raw (NO EB), single c + alpha ===")
    c2, alpha2, nll2, cg2, ag2, _, _ = fit_staged(
        Xtr, ytr, btr, alpha_init=np.full(10, 0.01), scale_init=1.0)
    print(f"  c={c2:.6f}, ||alpha||={np.linalg.norm(alpha2):.6f}")
    print(f"  c-grad = {cg2:.4e}, alpha-grad pos={int((ag2>0).sum())}/10")
    print(f"  alpha = {np.round(alpha2, 5)}")
    test_lam2 = c2 * bte + Xte @ alpha2
    nll_te2 = evaluate_count_forecast(yte, np.clip(test_lam2, 1e-8, None))["mean_poisson_nll"]
    print(f"  TEST NLL = {nll_te2:.4f}")

    # ----------------------------------------------------------------
    # 4. Substitute alpha_joint into staged objective (base = mu_eb * b_t),
    #    re-fit only c, see whether loss changes.
    # ----------------------------------------------------------------
    print("\n=== [DIAGNOSTIC] staged objective with alpha=alpha_joint, sweep c ===")
    cs = np.linspace(0.5, 1.3, 17)
    losses_aj = [staged_loss_grad(float(c), alpha_j, Xtr, ytr, base_eb_train, 1e-4, 10.0)[0] for c in cs]
    losses_a0 = [staged_loss_grad(float(c), np.zeros(10), Xtr, ytr, base_eb_train, 1e-4, 10.0)[0] for c in cs]
    bj = int(np.argmin(losses_aj)); b0 = int(np.argmin(losses_a0))
    print(f"  best (alpha=alpha_joint): c={cs[bj]:.3f}, loss={losses_aj[bj]:.4f}")
    print(f"  best (alpha=0)          : c={cs[b0]:.3f}, loss={losses_a0[b0]:.4f}")
    print(f"  delta = {losses_aj[bj] - losses_a0[b0]:+.4f}  "
          f"(positive => alpha_joint hurts staged loss when base = mu_eb * b_t)")

    # ----------------------------------------------------------------
    # 5. The reverse direction: what if we keep alpha=alpha_joint and go
    #    back to RAW b_t base, then optimize c? Should match the joint loss
    #    (without per-user lambda_u, just one global scalar).
    # ----------------------------------------------------------------
    print("\n=== [DIAGNOSTIC] sweep c on raw b_t at alpha=0 vs alpha=alpha_joint ===")
    losses_raw_a0 = [staged_loss_grad(float(c), np.zeros(10), Xtr, ytr, btr, 1e-4, 10.0)[0] for c in cs]
    losses_raw_aj = [staged_loss_grad(float(c), alpha_j, Xtr, ytr, btr, 1e-4, 10.0)[0] for c in cs]
    bj2 = int(np.argmin(losses_raw_aj)); b02 = int(np.argmin(losses_raw_a0))
    print(f"  best on raw b_t (alpha=0)         : c={cs[b02]:.3f}, loss={losses_raw_a0[b02]:.4f}")
    print(f"  best on raw b_t (alpha=alpha_joint): c={cs[bj2]:.3f}, loss={losses_raw_aj[bj2]:.4f}")
    print(f"  delta = {losses_raw_aj[bj2] - losses_raw_a0[b02]:+.4f}  "
          f"(if negative, alpha_joint helps when base is RAW)")

    # ----------------------------------------------------------------
    # 6. Compare: are mu_eb predictions and joint lambda_u close?
    # ----------------------------------------------------------------
    print("\n=== Per-user comparison: mu_eb vs lambda_u (joint) ===")
    mu_per_user = np.array([mutr[uidtr == u].mean() for u in unique_uids])
    print(f"  mu_eb:   mean={mu_per_user.mean():.4f}, q05={np.quantile(mu_per_user, 0.05):.4f}, "
          f"q50={np.quantile(mu_per_user, 0.5):.4f}, q95={np.quantile(mu_per_user, 0.95):.4f}")
    print(f"  lambda_u:mean={lam_u.mean():.4f}, q05={np.quantile(lam_u, 0.05):.4f}, "
          f"q50={np.quantile(lam_u, 0.5):.4f}, q95={np.quantile(lam_u, 0.95):.4f}")
    correlation = np.corrcoef(mu_per_user, lam_u)[0, 1]
    print(f"  corr(mu_eb, lambda_u) = {correlation:.4f}")


if __name__ == "__main__":
    main()

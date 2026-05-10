"""Sensitivity of Personalized Gamma-Poisson on 14d blocks to (α, β) constraints.

Tests whether the bug is specifically about marginal MLE picking α < 1 (which
makes the Gamma improper-at-zero), by trying:

  (1) refit-EB                         — current ch.10, marginal MLE on 14d
  (2) frozen 207d                      — α=0.8774, β=0.8808 (transfer)
  (3) refit α ≥ 1                       — constrained marginal MLE
  (4) fixed α=1, β refit                — exponential prior
  (5) fixed α=β=0.5                     — α<1 manually
  (6) fixed α=β=2                       — α>1 manually (well-behaved Gamma)
"""

from __future__ import annotations

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

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.diploma_baselines.data import load_daily_grid
from src.diploma_baselines.models import GlobalRollingSeasonalPoissonModel


TARGET_COL = "to_ord"
WINDOW_SIZE = 7
CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14


def neg_log_marginal(alpha: float, beta: float, y: np.ndarray, exposure: np.ndarray) -> float:
    return -float(np.sum(
        gammaln(y + alpha)
        - gammaln(alpha)
        - gammaln(y + 1.0)
        + alpha * np.log(beta)
        + y * np.log(np.clip(exposure, 1e-12, None))
        - (y + alpha) * np.log(beta + exposure)
    ))


def fit_eb_constrained(y, exposure, alpha_min=None, alpha_fixed=None, beta_fixed=None):
    """Fit marginal MLE with optional constraints."""
    if alpha_fixed is not None and beta_fixed is not None:
        return alpha_fixed, beta_fixed

    if alpha_fixed is not None:
        # Fit only β
        def _obj(log_beta):
            return neg_log_marginal(alpha_fixed, float(np.exp(log_beta[0])), y, exposure)
        res = minimize(_obj, x0=np.array([0.0]), method="L-BFGS-B")
        beta = float(np.exp(res.x[0]))
        return alpha_fixed, beta

    # Fit both with optional alpha lower bound
    def _obj(log_params):
        a = float(np.exp(log_params[0]))
        b = float(np.exp(log_params[1]))
        return neg_log_marginal(a, b, y, exposure)

    if alpha_min is None:
        bounds = None
    else:
        bounds = [(np.log(alpha_min), 10.0), (-10.0, 10.0)]
    res = minimize(_obj, x0=np.log([2.0, 2.0]), method="L-BFGS-B", bounds=bounds)
    return float(np.exp(res.x[0])), float(np.exp(res.x[1]))


from src.diploma_baselines.metrics import evaluate_count_forecast


def standard_poisson_nll(y: np.ndarray, lam: np.ndarray) -> float:
    """Full Poisson NLL via evaluate_count_forecast (includes log(y!) term)."""
    return float(evaluate_count_forecast(np.asarray(y, dtype=float), np.asarray(lam, dtype=float))["mean_poisson_nll"])


def predict_eb(y_train_per_user: pd.Series, exposure_per_user: pd.Series, alpha: float, beta: float, test_uids: np.ndarray, base_test: np.ndarray) -> np.ndarray:
    mu_post = (alpha + y_train_per_user) / (beta + exposure_per_user)
    prior_mean = alpha / beta
    mu_for_test = pd.Series(test_uids).map(mu_post).fillna(prior_mean).to_numpy(dtype=float)
    return mu_for_test * base_test


SETTINGS = [
    ("(1) refit α≥0 (current)", dict()),
    ("(2) frozen 207d (α=0.8774,β=0.8808)", dict(alpha_fixed=0.8774, beta_fixed=0.8808)),
    ("(3) refit with α≥1", dict(alpha_min=1.0)),
    ("(4) α=1 fixed, β refit", dict(alpha_fixed=1.0)),
    ("(5) α=β=0.5 fixed", dict(alpha_fixed=0.5, beta_fixed=0.5)),
    ("(6) α=β=2 fixed", dict(alpha_fixed=2.0, beta_fixed=2.0)),
]


def main() -> None:
    print("Loading data...")
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=[TARGET_COL],
    )
    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    blocks = []
    cursor = CV_GLOBAL_START
    idx = 0
    while True:
        block_end = cursor + pd.Timedelta(days=BLOCK_LEN - 1)
        if block_end > CV_GLOBAL_END:
            break
        train_end = cursor + pd.Timedelta(days=TRAIN_LEN - 1)
        blocks.append({"block_idx": idx, "block_start": cursor, "train_end": train_end, "test_start": train_end + pd.Timedelta(days=1), "test_end": block_end})
        idx += 1
        cursor = block_end + pd.Timedelta(days=1)

    results = {label: [] for label, _ in SETTINGS}
    fitted_alphas = {label: [] for label, _ in SETTINGS}
    fitted_betas = {label: [] for label, _ in SETTINGS}

    for block in blocks:
        block_train_df = full_df.loc[
            (full_df["event_date"] >= block["block_start"]) & (full_df["event_date"] <= block["train_end"])
        ].copy()
        block_test_df = full_df.loc[
            (full_df["event_date"] >= block["test_start"]) & (full_df["event_date"] <= block["test_end"])
        ].copy()

        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs_block = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean, daily_mean_full
        )
        base_train = rs_block.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)
        base_test = rs_block.predict_for_dates(block_test_df["event_date"]).to_numpy(dtype=float)

        df_aug = block_train_df.copy()
        df_aug["base_lambda"] = base_train
        per_user = df_aug.groupby("user_id").agg(y_sum=(TARGET_COL, "sum"), exposure=("base_lambda", "sum"))
        y_arr = per_user["y_sum"].to_numpy(dtype=float)
        exp_arr = per_user["exposure"].to_numpy(dtype=float)

        test_uids = block_test_df["user_id"].to_numpy()
        y_test = block_test_df[TARGET_COL].to_numpy(dtype=float)

        for label, kwargs in SETTINGS:
            alpha, beta = fit_eb_constrained(y_arr, exp_arr, **kwargs)
            lam = predict_eb(per_user["y_sum"], per_user["exposure"], alpha, beta, test_uids, base_test)
            nll = standard_poisson_nll(y_test, lam)
            results[label].append(nll)
            fitted_alphas[label].append(alpha)
            fitted_betas[label].append(beta)

    print("\n=== Mean test NLL across 13 blocks ===\n")
    print(f"{'Setting':<45} {'Mean NLL':>10} {'Mean α':>9} {'Mean β':>9}")
    for label, _ in SETTINGS:
        nll_mean = float(np.mean(results[label]))
        a_mean = float(np.mean(fitted_alphas[label]))
        b_mean = float(np.mean(fitted_betas[label]))
        print(f"{label:<45} {nll_mean:>10.4f} {a_mean:>9.3f} {b_mean:>9.3f}")

    out_path = Path("diploma/reports/blockwise_cv/personalized_gp_alpha_bounds_14d.csv")
    rows = []
    for i, block in enumerate(blocks):
        for label, _ in SETTINGS:
            rows.append({
                "block_idx": block["block_idx"],
                "setting": label,
                "test_nll": results[label][i],
                "fitted_alpha": fitted_alphas[label][i],
                "fitted_beta": fitted_betas[label][i],
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

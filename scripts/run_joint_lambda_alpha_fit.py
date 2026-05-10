"""Joint MLE for personalized scale + Hawkes coefficients on chapter-6 protocol.

Current "staged" pipeline:
  1. Fit RS baseline b_t on train.
  2. Fit per-user multiplier mu_u via Empirical Bayes (Gamma-Poisson conjugate).
  3. Fit Hawkes c, alpha on top: lambda_t = c * mu_u * b_t + states_t @ alpha.

This script tries the "joint" variant:
  Fit lambda_u and alpha simultaneously by Poisson MLE:
      lambda_t = lambda_{u(t)} * b_t + states_t @ alpha
  with mild L2 shrinkage on (lambda_u - 1) to handle users with few obs,
  and the same alpha_l2 as in the staged setup.

Compares test NLL on chapter-6 split:
  Train: 2025-01-15 .. 2025-08-09  (~207 days, 1.99M user-days)
  Test:  2025-08-10 .. 2025-09-30  (~52 days,  0.51M user-days)
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

from src.diploma_baselines.data import filter_date_range, load_daily_grid, split_panel_by_date
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_user_states_cache,
    fit_pooled_additive_multi_kernel_hawkes,
    predict_pooled_additive_multi_kernel_hawkes,
)


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

ANALYSIS_START = pd.Timestamp("2025-01-15")
ANALYSIS_END = pd.Timestamp("2025-09-30")
TRAIN_RATIO = 0.8

OUTPUT_DIR = Path("diploma/reports/joint_lambda_alpha")


def standard_poisson_nll(y: np.ndarray, lam: np.ndarray) -> float:
    """Full Poisson NLL via evaluate_count_forecast (includes log(y!) term)."""
    return float(evaluate_count_forecast(np.asarray(y, dtype=float), np.asarray(lam, dtype=float))["mean_poisson_nll"])


def fit_joint(
    user_idx: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    states: np.ndarray,
    n_users: int,
    n_alpha: int,
    lambda_l2: float,
    alpha_l2: float,
    lambda_init: np.ndarray | None = None,
    alpha_init: np.ndarray | None = None,
    max_iter: int = 300,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Backward-compat wrapper around `fit_joint_hawkes` from the canonical module.

    Returns the legacy `(lam_u, alpha, info_dict)` tuple. `n_alpha` and
    `verbose` are accepted but not used: `n_alpha` is inferred from `states`
    and per-iteration printing was only ever a debugging aid.
    """
    from src.diploma_baselines.models import fit_joint_hawkes

    res = fit_joint_hawkes(
        user_idx=user_idx,
        y=y,
        b=b,
        states=states,
        n_users=n_users,
        lambda_l2=lambda_l2,
        alpha_l2=alpha_l2,
        lambda_init=lambda_init,
        alpha_init=alpha_init,
        max_iter=max_iter,
    )
    info = {
        "converged": res.converged,
        "n_iter": res.n_iter,
        "final_nll": res.train_loss,
        "message": "ok" if res.converged else "not converged",
    }
    return res.lam_u, res.alpha, info


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    analysis_df = filter_date_range(full_df, start_date=ANALYSIS_START, end_date=ANALYSIS_END)
    split = split_panel_by_date(analysis_df, train_ratio=TRAIN_RATIO)
    print(f"  split_date = {split.split_date.date()}, train rows = {len(split.train):,}, test rows = {len(split.test):,}")

    train_daily_mean = split.train.groupby("event_date")[TARGET_COL].mean().sort_index()
    rs_model = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean, daily_mean_full
    )

    base_train = rs_model.predict_for_dates(split.train["event_date"]).to_numpy(dtype=float)
    base_test = rs_model.predict_for_dates(split.test["event_date"]).to_numpy(dtype=float)

    # === Hawkes states (warmed up from full history) ===
    print("\nBuilding Hawkes states from full history...")
    cache = build_user_states_cache(full_df, features=HAWKES_FEATURES, half_lives=HALF_LIVES)
    n_alpha = cache.n_alpha
    print(f"  {len(cache.states_per_user):,} users, n_alpha = {n_alpha}")

    print("\nGathering states for train and test...")
    states_train = cache.gather_for(split.train)
    states_test = cache.gather_for(split.test)
    print(f"  states_train shape = {states_train.shape}, states_test shape = {states_test.shape}")

    # === Build user index 0..n_users-1 from sorted unique IDs in TRAIN
    train_uids = split.train["user_id"].to_numpy()
    test_uids = split.test["user_id"].to_numpy()
    unique_train_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
    n_users = int(len(unique_train_uids))
    uid_to_idx = {int(u): i for i, u in enumerate(unique_train_uids)}
    test_user_idx = np.array([uid_to_idx.get(int(u), -1) for u in test_uids], dtype=np.int64)
    print(f"  n_users in train = {n_users}, test users mapped = {(test_user_idx != -1).sum():,} / {len(test_user_idx):,}")

    y_train = split.train[TARGET_COL].to_numpy(dtype=float)
    y_test = split.test[TARGET_COL].to_numpy(dtype=float)

    results = {}

    # === Setup A: Staged EB + Hawkes (chapter 6 reference) ===
    print("\n=== Setup A: Staged EB + Hawkes (chapter 6 reference) ===")
    t0 = time.time()
    scaler = PersonalizedGammaPoissonScaler().fit(
        train_uids, y_train, base_train,
    )
    pers_train = scaler.predict(train_uids, base_train, method="posterior_mean")
    pers_test = scaler.predict(test_uids, base_test, method="posterior_mean")

    # Hawkes per-user blocks for the staged fit
    train_state_blocks = []
    train_y_blocks = []
    train_base_blocks = []
    for uid, idx_in_block in split.train.groupby("user_id", sort=False).indices.items():
        train_state_blocks.append(states_train[idx_in_block])
        train_y_blocks.append(y_train[idx_in_block])
        train_base_blocks.append(pers_train[idx_in_block])

    hawkes_staged = fit_pooled_additive_multi_kernel_hawkes(
        state_blocks=train_state_blocks,
        y_blocks=train_y_blocks,
        base_blocks=train_base_blocks,
        half_lives=HALF_LIVES,
        feature_names=HAWKES_FEATURES,
        alpha_l2=1e-4,
        learn_base_scale=True,
        scale_l2=10.0,
        scale_init=1.0,
        max_iter=300,
    )
    nll_a_pers_only = standard_poisson_nll(y_test, pers_test)
    lam_a = hawkes_staged.base_scale * pers_test + states_test.astype(float) @ hawkes_staged.alpha
    nll_a = standard_poisson_nll(y_test, lam_a)
    print(f"  EB only test NLL = {nll_a_pers_only:.4f}")
    print(f"  EB+Hawkes test NLL = {nll_a:.4f}  c={hawkes_staged.base_scale:.4f}  ||α||={np.linalg.norm(hawkes_staged.alpha):.4f}")
    print(f"  ({time.time() - t0:.1f}s)")

    results["A_staged_EB_plus_Hawkes"] = {
        "test_nll": nll_a,
        "test_nll_eb_only": nll_a_pers_only,
        "c": float(hawkes_staged.base_scale),
        "alpha_norm": float(np.linalg.norm(hawkes_staged.alpha)),
        "alpha": hawkes_staged.alpha.tolist(),
    }

    # === Setup B: Joint fit with several lambda_l2 levels ===
    LAMBDA_L2_GRID = [0.0, 1.0, 10.0, 100.0]
    ALPHA_L2 = 1e-4

    for lam_l2 in LAMBDA_L2_GRID:
        print(f"\n=== Setup B (lambda_l2={lam_l2}, alpha_l2={ALPHA_L2}) ===")
        t0 = time.time()
        lam_u_fit, alpha_fit, info_fit = fit_joint(
            user_idx=train_user_idx,
            y=y_train,
            b=base_train,
            states=states_train.astype(float),
            n_users=n_users,
            n_alpha=n_alpha,
            lambda_l2=lam_l2,
            alpha_l2=ALPHA_L2,
            max_iter=400,
        )
        # Predict on test (test users that don't appear in train get lambda_u = 1.0 default)
        lam_u_for_test = np.where(test_user_idx >= 0, lam_u_fit[np.maximum(test_user_idx, 0)], 1.0)
        lam_b = lam_u_for_test * base_test + states_test.astype(float) @ alpha_fit
        nll_b = standard_poisson_nll(y_test, lam_b)
        alpha_norm = float(np.linalg.norm(alpha_fit))

        # Diagnostic statistics on lambda_u
        lam_stats = {
            "mean": float(np.mean(lam_u_fit)),
            "median": float(np.median(lam_u_fit)),
            "std": float(np.std(lam_u_fit)),
            "p05": float(np.percentile(lam_u_fit, 5)),
            "p95": float(np.percentile(lam_u_fit, 95)),
            "frac_at_lower_bound": float(np.mean(lam_u_fit <= 0.001 + 1e-8)),
        }

        print(
            f"  test NLL = {nll_b:.4f}  ||α||={alpha_norm:.4f}  "
            f"λ_u mean={lam_stats['mean']:.3f}  median={lam_stats['median']:.3f}  "
            f"std={lam_stats['std']:.3f}  [p5={lam_stats['p05']:.3f}, p95={lam_stats['p95']:.3f}]  "
            f"@lower={lam_stats['frac_at_lower_bound']:.1%}"
        )
        print(f"  iters={info_fit['n_iter']}  ({time.time() - t0:.1f}s)")

        results[f"B_joint_lambda_l2_{lam_l2}"] = {
            "test_nll": nll_b,
            "alpha_norm": alpha_norm,
            "alpha": alpha_fit.tolist(),
            "lambda_l2": lam_l2,
            "lambda_stats": lam_stats,
            "fit_info": info_fit,
        }

        # For the best setting (gamma=1) save a ladder-compatible summary
        if lam_l2 == 1.0:
            full_metrics = evaluate_count_forecast(y_test, lam_b)
            ladder_dir = Path("diploma/reports/joint_lambda_alpha_ch6")
            ladder_dir.mkdir(parents=True, exist_ok=True)
            with open(ladder_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "test_panel": {"rows": int(len(y_test))},
                        "test_metrics_joint_hawkes": full_metrics,
                        "fit_params": {
                            "lambda_l2": float(lam_l2),
                            "alpha_l2": float(ALPHA_L2),
                            "alpha_norm": float(alpha_norm),
                            "alpha": alpha_fit.tolist(),
                            "lambda_stats": lam_stats,
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    # === Reference: same lambda_l2 grid but with alpha frozen at 0 (no Hawkes, just MLE on lambda_u) ===
    for lam_l2 in LAMBDA_L2_GRID:
        print(f"\n=== Reference: Joint MLE only on lambda_u (lambda_l2={lam_l2}, alpha=0) ===")
        t0 = time.time()
        # Trick: pass states with all zeros to make alpha grad always tiny, and use alpha_l2=infinity-ish
        # Easier: just optimize only lambda_u manually with the same data
        # For simplicity fix alpha to zero: init alpha=0 and L2=very large

        def fit_lambda_only_with_baseline(lambda_l2_):
            user_idx_arr = train_user_idx
            init_lam = np.ones(n_users)
            bounds = [(0.001, 50.0)] * n_users

            def _obj(params):
                mu = np.clip(params[user_idx_arr] * base_train, 1e-8, None)
                nll = float(
                    np.sum(mu - y_train * np.log(mu))
                    + lambda_l2_ * np.sum((params - 1.0) ** 2)
                )
                residual = 1.0 - y_train / mu
                grad = np.zeros(n_users)
                np.add.at(grad, user_idx_arr, residual * base_train)
                grad += 2.0 * lambda_l2_ * (params - 1.0)
                return nll, grad

            res = minimize(_obj, init_lam, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 400})
            return res.x

        lam_u_only = fit_lambda_only_with_baseline(lam_l2)
        lam_for_test = np.where(test_user_idx >= 0, lam_u_only[np.maximum(test_user_idx, 0)], 1.0)
        lam = lam_for_test * base_test
        nll_ref = standard_poisson_nll(y_test, lam)
        print(f"  test NLL (lambda only, no Hawkes) = {nll_ref:.4f}  ({time.time() - t0:.1f}s)")
        results[f"REF_lambda_only_l2_{lam_l2}"] = {
            "test_nll": nll_ref,
            "lambda_l2": lam_l2,
        }

        # For γ=1 — save ladder-compatible summary (Personalized L2 in chapter 8)
        if lam_l2 == 1.0:
            full_metrics = evaluate_count_forecast(y_test, lam)
            ladder_dir = Path("diploma/reports/personalized_l2_ch6")
            ladder_dir.mkdir(parents=True, exist_ok=True)
            with open(ladder_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "test_panel": {"rows": int(len(y_test))},
                        "test_metrics_personalized_l2": full_metrics,
                        "fit_params": {
                            "lambda_l2": float(lam_l2),
                            "n_users": int(n_users),
                            "lambda_mean": float(np.mean(lam_u_only)),
                            "lambda_median": float(np.median(lam_u_only)),
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # === Plot 1: test NLL by lambda_l2 (joint + ref) ===
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    nll_a_val = results["A_staged_EB_plus_Hawkes"]["test_nll"]
    nll_a_pers = results["A_staged_EB_plus_Hawkes"]["test_nll_eb_only"]
    ax.axhline(nll_a_val, color="#0B3C5D", linewidth=2.0, linestyle="-", label=f"A: staged EB+Hawkes ({nll_a_val:.4f})")
    ax.axhline(nll_a_pers, color="#0B3C5D", linewidth=1.0, linestyle=":", label=f"A: EB only, no Hawkes ({nll_a_pers:.4f})")

    xs_grid = np.array(LAMBDA_L2_GRID, dtype=float)
    xs_log = np.where(xs_grid == 0, 0.1, xs_grid)
    nll_b = [results[f"B_joint_lambda_l2_{l2}"]["test_nll"] for l2 in LAMBDA_L2_GRID]
    nll_ref = [results[f"REF_lambda_only_l2_{l2}"]["test_nll"] for l2 in LAMBDA_L2_GRID]
    ax.plot(xs_log, nll_b, "o-", color="#D2691E", label="B: joint λ_u + α", linewidth=1.8, markersize=8)
    ax.plot(xs_log, nll_ref, "s--", color="#888888", label="REF: joint λ_u only (no Hawkes)", linewidth=1.4, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("L2 shrinkage on (λ_u − 1)")
    ax.set_ylabel("Test NLL per user-day")
    ax.set_title("Joint λ_u + α MLE vs staged EB+Hawkes (chapter-6 protocol)")
    ax.set_xticks(xs_log)
    ax.set_xticklabels([f"{x:g}" if x != 0.1 else "0" for x in xs_log])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "joint_vs_staged_nll.png", dpi=150)
    plt.close(fig)

    # === Plot 2: alpha_norm by lambda_l2 (joint) ===
    alpha_norms_b = [results[f"B_joint_lambda_l2_{l2}"]["alpha_norm"] for l2 in LAMBDA_L2_GRID]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.axhline(results["A_staged_EB_plus_Hawkes"]["alpha_norm"], color="#0B3C5D", linewidth=1.5, linestyle="-",
               label=f"A: staged ||α||={results['A_staged_EB_plus_Hawkes']['alpha_norm']:.4f}")
    ax.plot(xs_log, alpha_norms_b, "o-", color="#D2691E", linewidth=1.8, markersize=8, label="B: joint ||α||")
    ax.set_xscale("log")
    ax.set_xlabel("L2 shrinkage on (λ_u − 1)")
    ax.set_ylabel("Fitted ||α||_2")
    ax.set_title("Hawkes ||α||_2 in joint fit at different λ_u shrinkage levels")
    ax.set_xticks(xs_log)
    ax.set_xticklabels([f"{x:g}" if x != 0.1 else "0" for x in xs_log])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "joint_alpha_norm.png", dpi=150)
    plt.close(fig)

    print("\nFinal summary:")
    print(f"  A: Staged EB+Hawkes:                NLL={nll_a_val:.4f}  ||α||={results['A_staged_EB_plus_Hawkes']['alpha_norm']:.4f}")
    for l2 in LAMBDA_L2_GRID:
        bb = results[f"B_joint_lambda_l2_{l2}"]
        rr = results[f"REF_lambda_only_l2_{l2}"]
        print(
            f"  joint l2={l2:>5g}: λ_u+α NLL={bb['test_nll']:.4f}  ||α||={bb['alpha_norm']:.4f}  |  "
            f"λ_u only NLL={rr['test_nll']:.4f}"
        )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

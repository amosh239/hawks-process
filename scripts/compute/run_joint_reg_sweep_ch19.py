"""Chapter 19: Joint Hawkes regularization sweep on 207d main split.

Sweep `lambda_l2 ∈ [0, 5]` over a wide grid; for each value × each of 3
channels (`searches`, `to_cart`, `to_ord`) fit Joint Hawkes on the same
207d train (`2025-01-15 .. 2025-08-09`) and evaluate on 52d test
(`2025-08-10 .. 2025-09-30`). Half-life = 1 day, `α_l2 = 1e-4`.

Per-(λ_l2, target) record:
  * α (3-vector)
  * mean(λ_u), mean per-row baseline contribution `λ_u · b_t`
  * mean per-row Hawkes contribution `α^T s`
  * train and test NLL

Plots:
  1. trade-off cloud — per-channel scatter of (mean baseline, mean Hawkes)
     coloured by λ_l2; shows how mass migrates from baseline into Hawkes
     as regularization tightens.
  2. test NLL vs λ_l2 — one line per channel.
  3. 3×3 grid of α[i,j] vs λ_l2 — how each Hawkes coefficient moves with
     regularization.

Uses ProcessPoolExecutor for parallel fits (~4–6× speedup on Mac).
"""

from __future__ import annotations

# Limit BLAS threads BEFORE numpy import — required for proper parallelism
# across multiple processes (otherwise each worker oversubscribes the CPU).
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from src.diploma_baselines.data import filter_date_range, load_daily_grid
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_user_states_cache,
    fit_joint_hawkes,
)


CHANNELS = ("searches", "to_cart", "to_ord")
HALF_LIVES = (1.0,)
WINDOW_SIZE = 7
ALPHA_L2 = 1e-4
MAX_ITER = 500
N_WORKERS = 6

LAMBDA_L2_GRID = (0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0)

TRAIN_START = pd.Timestamp("2025-01-15")
TRAIN_END = pd.Timestamp("2025-08-09")
TEST_START = pd.Timestamp("2025-08-10")
TEST_END = pd.Timestamp("2025-09-30")

OUT_DIR = Path("diploma/reports/19_joint_reg_sweep")


# Per-channel data prepared in main, attached as module globals via init_worker
# so every worker process sees the same arrays without re-pickling per job.
_GLOBAL: dict = {}


def init_worker(payload: dict) -> None:
    global _GLOBAL
    _GLOBAL = payload


def worker_fit(job: tuple) -> dict:
    """Single Joint Hawkes fit for (λ_l2, target). Reads shared arrays from `_GLOBAL`."""
    lambda_l2, ch_idx, target = job
    g = _GLOBAL

    y_train = g["y_train"][ch_idx]
    y_test = g["y_test"][ch_idx]
    base_train = g["base_train"][ch_idx]
    base_test = g["base_test"][ch_idx]
    pers_train = g["pers_train"][ch_idx]
    pers_test = g["pers_test"][ch_idx]
    train_user_idx = g["train_user_idx"]
    test_user_idx = g["test_user_idx"]
    n_users = g["n_users"]
    states_train = g["states_train"]
    states_test = g["states_test"]

    t0 = time.time()
    j = fit_joint_hawkes(
        user_idx=train_user_idx,
        y=y_train,
        b=base_train,
        states=states_train.astype(float),
        n_users=n_users,
        lambda_l2=float(lambda_l2),
        alpha_l2=ALPHA_L2,
        max_iter=MAX_ITER,
    )
    elapsed = time.time() - t0

    lam_u = j.lam_u
    alpha = j.alpha

    # --- Train metrics (and baseline / Hawkes contribution decomposition)
    lam_u_train_per_row = lam_u[train_user_idx]
    baseline_train_per_row = lam_u_train_per_row * base_train
    hawkes_train_per_row = states_train.astype(float) @ alpha
    lam_train = np.clip(baseline_train_per_row + hawkes_train_per_row, 1e-8, None)
    hawkes_train_nll = float(evaluate_count_forecast(y_train, lam_train)["mean_poisson_nll"])
    personalized_train_nll = float(evaluate_count_forecast(y_train, pers_train)["mean_poisson_nll"])

    # --- Test metrics
    lam_u_test_per_row = np.where(
        test_user_idx >= 0, lam_u[np.maximum(test_user_idx, 0)], 1.0
    )
    baseline_test_per_row = lam_u_test_per_row * base_test
    hawkes_test_per_row = states_test.astype(float) @ alpha
    lam_test = np.clip(baseline_test_per_row + hawkes_test_per_row, 1e-8, None)
    hawkes_test_nll = float(evaluate_count_forecast(y_test, lam_test)["mean_poisson_nll"])
    personalized_test_nll = float(evaluate_count_forecast(y_test, pers_test)["mean_poisson_nll"])

    return {
        "lambda_l2": float(lambda_l2),
        "ch_idx": int(ch_idx),
        "target": target,
        "alpha": np.asarray(alpha, dtype=float),
        "alpha_norm": float(np.linalg.norm(alpha)),
        "lam_u_mean": float(lam_u.mean()),
        "lam_u_floor_share": float((lam_u <= 0.001 + 1e-6).mean()),
        "mean_baseline_train": float(baseline_train_per_row.mean()),
        "mean_hawkes_train": float(hawkes_train_per_row.mean()),
        "mean_baseline_test": float(baseline_test_per_row.mean()),
        "mean_hawkes_test": float(hawkes_test_per_row.mean()),
        "personalized_train_nll": personalized_train_nll,
        "personalized_test_nll": personalized_test_nll,
        "hawkes_train_nll": hawkes_train_nll,
        "hawkes_test_nll": hawkes_test_nll,
        "delta_train": hawkes_train_nll - personalized_train_nll,
        "delta_test": hawkes_test_nll - personalized_test_nll,
        "converged": bool(j.converged),
        "n_iter": int(j.n_iter),
        "elapsed_s": float(elapsed),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_ch = len(CHANNELS)
    n_lam = len(LAMBDA_L2_GRID)

    print("Loading data...")
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=list(CHANNELS),
    )
    daily_mean_full = {
        ch: full_df.groupby("event_date")[ch].mean().sort_index() for ch in CHANNELS
    }

    train_df = filter_date_range(full_df, start_date=TRAIN_START, end_date=TRAIN_END).copy()
    test_df = filter_date_range(full_df, start_date=TEST_START, end_date=TEST_END).copy()
    print(f"  train rows: {len(train_df):,}, test rows: {len(test_df):,}")

    print("Building Hawkes states cache...")
    cache = build_user_states_cache(full_df, features=CHANNELS, half_lives=HALF_LIVES)
    states_train = cache.gather_for(train_df)
    states_test = cache.gather_for(test_df)
    print(f"  states_train shape = {states_train.shape}")

    # === Per-channel baselines ===
    print("Building per-channel rolling-seasonal baselines and EB priors...")
    train_uids = train_df["user_id"].to_numpy()
    test_uids = test_df["user_id"].to_numpy()
    unique_train_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
    n_users = int(len(unique_train_uids))
    uid_to_idx = {int(u): i for i, u in enumerate(unique_train_uids)}
    test_user_idx = np.array(
        [uid_to_idx.get(int(u), -1) for u in test_uids], dtype=np.int64
    )

    y_train = []
    y_test = []
    base_train = []
    base_test = []
    pers_train = []
    pers_test = []
    for ch in CHANNELS:
        y_tr = train_df[ch].to_numpy(dtype=float)
        y_te = test_df[ch].to_numpy(dtype=float)
        train_daily_mean = train_df.groupby("event_date")[ch].mean().sort_index()
        rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            train_daily_mean, daily_mean_full[ch],
        )
        b_tr = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
        b_te = rs.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)
        scaler = PersonalizedGammaPoissonScaler().fit(train_uids, y_tr, b_tr)
        p_tr = scaler.predict(train_uids, b_tr, method="posterior_mean")
        p_te = scaler.predict(test_uids, b_te, method="posterior_mean")
        y_train.append(y_tr); y_test.append(y_te)
        base_train.append(b_tr); base_test.append(b_te)
        pers_train.append(p_tr); pers_test.append(p_te)
        print(f"  {ch}: y_train mean = {y_tr.mean():.4f}")

    payload = {
        "y_train": y_train, "y_test": y_test,
        "base_train": base_train, "base_test": base_test,
        "pers_train": pers_train, "pers_test": pers_test,
        "train_user_idx": train_user_idx, "test_user_idx": test_user_idx,
        "n_users": n_users,
        "states_train": states_train, "states_test": states_test,
    }

    # === Build job list and dispatch in parallel ===
    jobs = [(lam, i, t) for lam in LAMBDA_L2_GRID for i, t in enumerate(CHANNELS)]
    print(f"\nLaunching {len(jobs)} fits across {N_WORKERS} workers...")
    t_global = time.time()

    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=N_WORKERS, initializer=init_worker, initargs=(payload,),
    ) as ex:
        futures = [ex.submit(worker_fit, job) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            rows.append(res)
            print(
                f"  λ_l2={res['lambda_l2']:>5.2f} {res['target']:<10s} "
                f"||α||={res['alpha_norm']:.4f} "
                f"mean λ_u={res['lam_u_mean']:.4f} (floor={res['lam_u_floor_share']*100:5.1f}%) "
                f"Δtrain={res['delta_train']:+.4f} Δtest={res['delta_test']:+.4f}  "
                f"({res['elapsed_s']:.1f}s, n_iter={res['n_iter']}, ok={res['converged']})"
            )
    print(f"\nAll fits done in {time.time() - t_global:.1f}s wall.")

    # === Aggregate into arrays for plotting ===
    rows.sort(key=lambda r: (r["lambda_l2"], r["ch_idx"]))

    alpha_cube = np.zeros((n_lam, n_ch, n_ch))   # [lam, target, source]
    test_nll = np.zeros((n_lam, n_ch))
    train_nll = np.zeros((n_lam, n_ch))
    delta_test = np.zeros((n_lam, n_ch))
    delta_train = np.zeros((n_lam, n_ch))
    mean_baseline_train = np.zeros((n_lam, n_ch))
    mean_hawkes_train = np.zeros((n_lam, n_ch))
    lam_u_mean = np.zeros((n_lam, n_ch))
    lam_u_floor_share = np.zeros((n_lam, n_ch))

    for r in rows:
        li = LAMBDA_L2_GRID.index(r["lambda_l2"])
        ci = r["ch_idx"]
        alpha_cube[li, ci, :] = r["alpha"]
        test_nll[li, ci] = r["hawkes_test_nll"]
        train_nll[li, ci] = r["hawkes_train_nll"]
        delta_test[li, ci] = r["delta_test"]
        delta_train[li, ci] = r["delta_train"]
        mean_baseline_train[li, ci] = r["mean_baseline_train"]
        mean_hawkes_train[li, ci] = r["mean_hawkes_train"]
        lam_u_mean[li, ci] = r["lam_u_mean"]
        lam_u_floor_share[li, ci] = r["lam_u_floor_share"]

    # === Persist long-format CSV ===
    long_rows = []
    for r in rows:
        for src_idx, src in enumerate(CHANNELS):
            long_rows.append({
                "lambda_l2": r["lambda_l2"],
                "target": r["target"],
                "source": src,
                "alpha": float(r["alpha"][src_idx]),
                "lam_u_mean": r["lam_u_mean"],
                "mean_baseline_train": r["mean_baseline_train"],
                "mean_hawkes_train": r["mean_hawkes_train"],
                "hawkes_train_nll": r["hawkes_train_nll"],
                "hawkes_test_nll": r["hawkes_test_nll"],
                "delta_train": r["delta_train"],
                "delta_test": r["delta_test"],
            })
    pd.DataFrame(long_rows).to_csv(OUT_DIR / "sweep_results.csv", index=False)

    # === Plot 1: trade-off cloud (mean baseline vs mean Hawkes per row) ===
    fig, axes = plt.subplots(1, n_ch, figsize=(5.2 * n_ch, 5.0), squeeze=False)
    cmap = plt.cm.viridis
    norm_lam = plt.matplotlib.colors.Normalize(
        vmin=min(LAMBDA_L2_GRID), vmax=max(LAMBDA_L2_GRID)
    )
    for ci, target in enumerate(CHANNELS):
        ax = axes[0][ci]
        # connect the trajectory in order of increasing λ_l2
        ax.plot(
            mean_baseline_train[:, ci], mean_hawkes_train[:, ci],
            color="#888888", linewidth=0.8, alpha=0.6, zorder=1,
        )
        for li, lam in enumerate(LAMBDA_L2_GRID):
            ax.scatter(
                mean_baseline_train[li, ci], mean_hawkes_train[li, ci],
                color=cmap(norm_lam(lam)), s=120, edgecolors="white", linewidths=1.2,
                zorder=3,
            )
            ax.annotate(
                f"{lam:g}",
                (mean_baseline_train[li, ci], mean_hawkes_train[li, ci]),
                textcoords="offset points", xytext=(8, 4), fontsize=8,
            )
        ax.set_xlabel(r"mean baseline contribution  $\overline{\lambda_u \cdot b_t}$  (train)")
        ax.set_ylabel(r"mean Hawkes contribution  $\overline{\alpha^\top s}$  (train)")
        ax.set_title(f"target = {target}")
        ax.grid(linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_lam)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, label=r"$\lambda_{\ell_2}$")
    fig.suptitle(
        r"Trade-off: baseline mass vs Hawkes mass as $\lambda_{\ell_2}$ varies (per-channel)",
        fontsize=12,
    )
    fig.savefig(OUT_DIR / "tradeoff_cloud.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Plot 2: test NLL vs λ_l2  (1×3 row, tight y-axes, baseline as text) ===
    fig, axes = plt.subplots(1, n_ch, figsize=(4.0 * n_ch, 5.0), squeeze=False)
    ch_colors = {"searches": "#2E5EAA", "to_cart": "#7B3FAA", "to_ord": "#D2691E"}
    for ci, target in enumerate(CHANNELS):
        ax = axes[0][ci]
        vals = test_nll[:, ci]
        ax.plot(LAMBDA_L2_GRID, vals,
                marker="o", linewidth=1.8, color=ch_colors[target], label="Joint Hawkes")
        i_min = int(np.argmin(vals))
        ax.scatter([LAMBDA_L2_GRID[i_min]], [vals[i_min]],
                   s=120, facecolors="none", edgecolors=ch_colors[target],
                   linewidths=2.0, zorder=4, label=f"min @ λ={LAMBDA_L2_GRID[i_min]:g}")

        rng = vals.max() - vals.min()
        pad = max(rng * 0.20, 0.0005)
        ax.set_ylim(vals.min() - pad, vals.max() + pad)

        pers_const = float(
            evaluate_count_forecast(y_test[ci], pers_test[ci])["mean_poisson_nll"]
        )
        ax.text(0.02, 0.97,
                f"Personalized GP baseline:\n  {pers_const:.4f} (gap = {vals.min() - pers_const:+.4f})",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#666",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(0, max(LAMBDA_L2_GRID) * 1.1)
        ax.set_xlabel(r"$\lambda_{\ell_2}$")
        ax.set_ylabel(f"test NLL ({target})")
        ax.set_title(f"target = {target}", color=ch_colors[target], fontweight="bold")
        ax.grid(linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=8)

    fig.suptitle(
        r"Test NLL vs Joint Hawkes $\lambda_{\ell_2}$ (207d train) — tight y-axis, baseline as text",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "test_nll_vs_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Plot 3: 3x3 grid of α[i, j] vs λ_l2 ===
    fig, axes = plt.subplots(n_ch, n_ch, figsize=(3.4 * n_ch, 2.8 * n_ch), squeeze=False)
    for i, target in enumerate(CHANNELS):
        for j, src in enumerate(CHANNELS):
            ax = axes[i][j]
            vals = alpha_cube[:, i, j]
            ax.plot(LAMBDA_L2_GRID, vals, marker="o", linewidth=1.8, color="#7B3FAA")
            on_diag = i == j
            ax.set_title(
                f"{target} ← {src}" + ("  (self)" if on_diag else ""),
                fontsize=10, fontweight=("bold" if on_diag else "normal"),
            )
            ax.set_xscale("symlog", linthresh=0.01)
            ax.set_xlim(0, max(LAMBDA_L2_GRID) * 1.1)
            ax.grid(linestyle=":", alpha=0.5)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if i == n_ch - 1:
                ax.set_xlabel(r"$\lambda_{\ell_2}$")
            if j == 0:
                ax.set_ylabel(r"$\alpha_{i,j}$")
            ax.axhline(0, color="#888888", linewidth=0.6)
    fig.suptitle(
        r"Joint Hawkes $\alpha_{i,j}$ vs regularization $\lambda_{\ell_2}$  (207d train)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_vs_lambda_grid.png", dpi=150)
    plt.close(fig)

    # === Summary JSON ===
    summary = {
        "lambda_l2_grid": list(LAMBDA_L2_GRID),
        "channels": list(CHANNELS),
        "half_lives": list(HALF_LIVES),
        "alpha_l2": ALPHA_L2,
        "max_iter": MAX_ITER,
        "n_workers": N_WORKERS,
        "train_window": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "test_window": [str(TEST_START.date()), str(TEST_END.date())],
        "alpha_cube": alpha_cube.tolist(),
        "test_nll": test_nll.tolist(),
        "train_nll": train_nll.tolist(),
        "delta_test": delta_test.tolist(),
        "delta_train": delta_train.tolist(),
        "mean_baseline_train": mean_baseline_train.tolist(),
        "mean_hawkes_train": mean_hawkes_train.tolist(),
        "lam_u_mean": lam_u_mean.tolist(),
        "lam_u_floor_share": lam_u_floor_share.tolist(),
        "rows": [{**r, "alpha": r["alpha"].tolist()} for r in rows],
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()

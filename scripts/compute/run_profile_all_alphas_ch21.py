"""Chapter 21: Profile-likelihood sweep over each of 9 Hawkes-α coefficients.

For each (target, source) pair (3 × 3 = 9 coefficients):
  1. Cold-fit unfixed Joint Hawkes for `target` at λ_l2 = 1 → (λ̂_u, α̂)
  2. Build pin grid of 11 points around the unfixed value:
        grid = linspace(0, max(0.10, 2 · α̂_source), 11)
  3. Pin α[target ← source] to each grid value;
     re-optimize remaining params (λ_u and the other 2 α-coefs);
     warm-start sweep + 2 polish passes.
  4. Record train NLL and test NLL.

Outputs:
  - train_nll_grid.png — 3×3 grid (row = target, col = source)
  - test_nll_grid.png  — 3×3 grid
  - sweep_results.csv  — long-format table
  - summary.json       — per-coefficient eigvals, anchor, pin grid, NLLs
"""

from __future__ import annotations

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
from scipy.optimize import minimize

from src.diploma_baselines.data import filter_date_range, load_daily_grid
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    GlobalRollingSeasonalPoissonModel,
    build_user_states_cache,
    fit_joint_hawkes,
)


CHANNELS = ("searches", "to_cart", "to_ord")
HALF_LIVES = (1.0,)
WINDOW_SIZE = 7
ALPHA_L2 = 1e-4
LAMBDA_L2 = 1.0

GRID_SIZE = 11
GRID_MIN_UPPER = 0.10              # ensure grid spans [0, 0.10] for tiny anchors
GRID_UPPER_MULT = 2.0              # else use [0, GRID_UPPER_MULT × anchor]

COLD_MAX_ITER = 3000
WARM_MAX_ITER = 1200
N_POLISH_PASSES = 2
N_WORKERS = 6

TRAIN_START = pd.Timestamp("2025-01-15")
TRAIN_END = pd.Timestamp("2025-08-09")
TEST_START = pd.Timestamp("2025-08-10")
TEST_END = pd.Timestamp("2025-09-30")

OUT_DIR = Path("diploma/reports/21_profile_all_alphas")


_GLOBAL: dict = {}


def init_worker(payload: dict) -> None:
    global _GLOBAL
    _GLOBAL = payload


def fit_with_pinned_alpha(
    user_idx, y, b, states, n_users, lambda_l2, alpha_l2,
    fixed_idx, fixed_value, max_iter,
    lam_init=None, alpha_free_init=None,
) -> dict:
    """L-BFGS-B fit of Joint Hawkes with α[fixed_idx] pinned to fixed_value."""
    n_alpha = int(states.shape[1])
    free_mask = np.ones(n_alpha, dtype=bool)
    free_mask[fixed_idx] = False
    free_indices = np.where(free_mask)[0]

    init_lam = np.ones(n_users, dtype=float) if lam_init is None else np.asarray(lam_init, dtype=float).copy()
    init_alpha_free = (
        np.full(n_alpha - 1, 0.01, dtype=float)
        if alpha_free_init is None
        else np.asarray(alpha_free_init, dtype=float).copy()
    )
    init = np.concatenate([init_lam, init_alpha_free])
    bounds = [(0.001, 50.0)] * n_users + [(0.0, 10.0)] * (n_alpha - 1)

    def fg(params):
        lam_u = params[:n_users]
        alpha_free = params[n_users:]
        alpha_full = np.empty(n_alpha, dtype=float)
        alpha_full[free_indices] = alpha_free
        alpha_full[fixed_idx] = fixed_value
        mu = np.clip(lam_u[user_idx] * b + states @ alpha_full, 1e-8, None)
        nll = float(
            np.sum(mu - y * np.log(mu))
            + lambda_l2 * np.sum((lam_u - 1.0) ** 2)
            + alpha_l2 * np.sum(alpha_full ** 2)
        )
        residual = 1.0 - y / mu
        grad_lam = np.zeros(n_users, dtype=np.float64)
        np.add.at(grad_lam, user_idx, residual * b)
        grad_lam += 2.0 * lambda_l2 * (lam_u - 1.0)
        grad_alpha_full = states.T @ residual + 2.0 * alpha_l2 * alpha_full
        grad_alpha_free = grad_alpha_full[free_indices]
        return nll, np.concatenate([grad_lam, grad_alpha_free])

    res = minimize(
        lambda p: fg(p)[0], init, method="L-BFGS-B",
        jac=lambda p: fg(p)[1], bounds=bounds, options={"maxiter": max_iter},
    )
    lam_u = res.x[:n_users]
    alpha_free = res.x[n_users:]
    alpha_full = np.empty(n_alpha, dtype=float)
    alpha_full[free_indices] = alpha_free
    alpha_full[fixed_idx] = fixed_value
    return {
        "lam_u": lam_u,
        "alpha_free": alpha_free,
        "alpha_full": alpha_full,
        "train_loss": float(res.fun),
        "converged": bool(res.success),
        "n_iter": int(getattr(res, "nit", 0)),
    }


def fit_one_point(
    pin_value, *, lam_init, alpha_free_init, max_iter,
    target_idx, fixed_idx,
):
    g = _GLOBAL
    y_tr = g["y_train"][target_idx]
    b_tr = g["base_train"][target_idx]
    user_idx = g["train_user_idx"]
    n_users = g["n_users"]
    states = g["states_train"]
    y_te = g["y_test"][target_idx]
    b_te = g["base_test"][target_idx]
    test_user_idx = g["test_user_idx"]
    states_test = g["states_test"]

    t0 = time.time()
    out = fit_with_pinned_alpha(
        user_idx=user_idx, y=y_tr, b=b_tr, states=states.astype(float),
        n_users=n_users, lambda_l2=LAMBDA_L2, alpha_l2=ALPHA_L2,
        fixed_idx=fixed_idx, fixed_value=float(pin_value), max_iter=max_iter,
        lam_init=lam_init, alpha_free_init=alpha_free_init,
    )
    elapsed = time.time() - t0
    lam_u = out["lam_u"]
    alpha_full = out["alpha_full"]

    # Train metrics
    lam_u_t = lam_u[user_idx]
    baseline_part = lam_u_t * b_tr
    hawkes_part = states.astype(float) @ alpha_full
    lam_t = np.clip(baseline_part + hawkes_part, 1e-8, None)
    train_nll_mean = float(evaluate_count_forecast(y_tr, lam_t)["mean_poisson_nll"])

    # Test metrics
    lam_u_te = np.where(
        test_user_idx >= 0, lam_u[np.maximum(test_user_idx, 0)], 1.0
    )
    baseline_te = lam_u_te * b_te
    hawkes_te = states_test.astype(float) @ alpha_full
    lam_te = np.clip(baseline_te + hawkes_te, 1e-8, None)
    test_nll_mean = float(evaluate_count_forecast(y_te, lam_te)["mean_poisson_nll"])

    return {
        "pin_value": float(pin_value),
        "alpha_full": alpha_full.tolist(),
        "lam_u": lam_u,
        "alpha_free": out["alpha_free"],
        "lam_u_mean": float(lam_u.mean()),
        "train_loss": out["train_loss"],
        "train_nll_mean": train_nll_mean,
        "test_nll_mean": test_nll_mean,
        "mean_baseline_train": float(baseline_part.mean()),
        "mean_hawkes_train": float(hawkes_part.mean()),
        "mean_baseline_test": float(baseline_te.mean()),
        "mean_hawkes_test": float(hawkes_te.mean()),
        "converged": out["converged"],
        "n_iter": out["n_iter"],
        "fit_time_s": elapsed,
    }


def worker_sweep(job):
    """Sweep one (target, source) pair: 11-point grid, warm-start + polish passes."""
    target_idx, source_idx, target, source, pin_grid, anchor_lam_u, anchor_alpha = job
    n_alpha = 3

    # Free indices = all except source_idx
    free_indices = [k for k in range(n_alpha) if k != source_idx]
    anchor_alpha_free = np.asarray([anchor_alpha[k] for k in free_indices], dtype=float)
    anchor_lam_u = np.asarray(anchor_lam_u, dtype=float)

    # Find anchor index in pin_grid (closest to unfixed value)
    anchor_pin_val = float(anchor_alpha[source_idx])
    anchor_idx = int(np.argmin(np.abs(np.asarray(pin_grid) - anchor_pin_val)))

    rows_by_pin: dict[float, dict] = {}

    # 1) Anchor — warm-start from unfixed solution
    pin_anchor = float(pin_grid[anchor_idx])
    res_anchor = fit_one_point(
        pin_anchor, lam_init=anchor_lam_u, alpha_free_init=anchor_alpha_free,
        max_iter=COLD_MAX_ITER, target_idx=target_idx, fixed_idx=source_idx,
    )
    rows_by_pin[pin_anchor] = res_anchor

    # 2) Sweep downwards
    prev = res_anchor
    for idx in range(anchor_idx - 1, -1, -1):
        pin_v = float(pin_grid[idx])
        r = fit_one_point(
            pin_v, lam_init=prev["lam_u"], alpha_free_init=prev["alpha_free"],
            max_iter=WARM_MAX_ITER, target_idx=target_idx, fixed_idx=source_idx,
        )
        rows_by_pin[pin_v] = r
        prev = r

    # 3) Sweep upwards
    prev = res_anchor
    for idx in range(anchor_idx + 1, len(pin_grid)):
        pin_v = float(pin_grid[idx])
        r = fit_one_point(
            pin_v, lam_init=prev["lam_u"], alpha_free_init=prev["alpha_free"],
            max_iter=WARM_MAX_ITER, target_idx=target_idx, fixed_idx=source_idx,
        )
        rows_by_pin[pin_v] = r
        prev = r

    # 4) Polish passes: re-anchor at current best, re-sweep
    for _ in range(N_POLISH_PASSES):
        losses = np.array([rows_by_pin[float(v)]["train_loss"] for v in pin_grid])
        best_idx = int(np.argmin(losses))
        prev = rows_by_pin[float(pin_grid[best_idx])]
        for idx in range(best_idx - 1, -1, -1):
            pin_v = float(pin_grid[idx])
            r = fit_one_point(
                pin_v, lam_init=prev["lam_u"], alpha_free_init=prev["alpha_free"],
                max_iter=WARM_MAX_ITER, target_idx=target_idx, fixed_idx=source_idx,
            )
            if r["train_loss"] < rows_by_pin[pin_v]["train_loss"] - 1e-6:
                rows_by_pin[pin_v] = r
            prev = r
        prev = rows_by_pin[float(pin_grid[best_idx])]
        for idx in range(best_idx + 1, len(pin_grid)):
            pin_v = float(pin_grid[idx])
            r = fit_one_point(
                pin_v, lam_init=prev["lam_u"], alpha_free_init=prev["alpha_free"],
                max_iter=WARM_MAX_ITER, target_idx=target_idx, fixed_idx=source_idx,
            )
            if r["train_loss"] < rows_by_pin[pin_v]["train_loss"] - 1e-6:
                rows_by_pin[pin_v] = r
            prev = r

    # Build serializable rows in pin_grid order
    rows = []
    for v in pin_grid:
        r = rows_by_pin[float(v)]
        rs = {k: r[k] for k in r if k not in ("lam_u", "alpha_free")}
        rows.append(rs)

    return {
        "target_idx": target_idx,
        "source_idx": source_idx,
        "target": target,
        "source": source,
        "pin_grid": list(pin_grid),
        "anchor_pin_val": anchor_pin_val,
        "rows": rows,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_ch = len(CHANNELS)

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
    states_train = cache.gather_for(train_df).astype(float)
    states_test = cache.gather_for(test_df).astype(float)
    print(f"  states_train shape = {states_train.shape}, states_test shape = {states_test.shape}")

    train_uids = train_df["user_id"].to_numpy()
    test_uids = test_df["user_id"].to_numpy()
    unique_train_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
    n_users = int(len(unique_train_uids))
    uid_to_idx = {int(u): i for i, u in enumerate(unique_train_uids)}
    test_user_idx = np.array(
        [uid_to_idx.get(int(u), -1) for u in test_uids], dtype=np.int64
    )

    print(f"  n_users = {n_users}")

    y_train, y_test = [], []
    base_train, base_test = [], []
    for ch in CHANNELS:
        y_tr = train_df[ch].to_numpy(dtype=float)
        y_te = test_df[ch].to_numpy(dtype=float)
        train_daily_mean = train_df.groupby("event_date")[ch].mean().sort_index()
        rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            train_daily_mean, daily_mean_full[ch],
        )
        b_tr = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
        b_te = rs.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)
        y_train.append(y_tr); y_test.append(y_te)
        base_train.append(b_tr); base_test.append(b_te)

    # === Cold-fit unfixed Joint Hawkes per target (anchors) ===
    print("\nCold-fitting unfixed Joint Hawkes per target...")
    anchors: list[dict] = []
    t_global = time.time()
    for ci, ch in enumerate(CHANNELS):
        t0 = time.time()
        j = fit_joint_hawkes(
            user_idx=train_user_idx, y=y_train[ci], b=base_train[ci],
            states=states_train, n_users=n_users,
            lambda_l2=LAMBDA_L2, alpha_l2=ALPHA_L2, max_iter=COLD_MAX_ITER,
        )
        anchors.append({
            "target": ch, "target_idx": ci,
            "lam_u": j.lam_u, "alpha": j.alpha,
            "converged": bool(j.converged), "n_iter": int(j.n_iter),
        })
        print(
            f"  {ch}: α = [{j.alpha[0]:.4f}, {j.alpha[1]:.4f}, {j.alpha[2]:.4f}]  "
            f"⟨λ_u⟩={j.lam_u.mean():.4f}  ({time.time()-t0:.1f}s, n_iter={j.n_iter}, ok={j.converged})"
        )
    print(f"  Anchors fit done in {time.time() - t_global:.1f}s.")

    payload = {
        "y_train": y_train, "y_test": y_test,
        "base_train": base_train, "base_test": base_test,
        "train_user_idx": train_user_idx, "test_user_idx": test_user_idx,
        "n_users": n_users,
        "states_train": states_train, "states_test": states_test,
    }

    # === Build grids and submit 9 sweep jobs ===
    print(f"\nBuilding pin grids and launching {n_ch * n_ch} sweep jobs across {N_WORKERS} workers...")
    jobs = []
    for a in anchors:
        target_idx = a["target_idx"]
        target = a["target"]
        for source_idx, source in enumerate(CHANNELS):
            anchor_val = float(a["alpha"][source_idx])
            upper = max(GRID_MIN_UPPER, GRID_UPPER_MULT * anchor_val)
            pin_grid = np.round(np.linspace(0.0, upper, GRID_SIZE), 6)
            jobs.append((
                target_idx, source_idx, target, source,
                tuple(float(v) for v in pin_grid),
                a["lam_u"], a["alpha"],
            ))
            print(f"  job: {target:<10s} ← {source:<10s}  anchor={anchor_val:.4f}  "
                  f"grid = [{pin_grid[0]:.4f}..{pin_grid[-1]:.4f}] step {pin_grid[1]-pin_grid[0]:.4f}")

    t_sweep = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=N_WORKERS, initializer=init_worker, initargs=(payload,),
    ) as ex:
        futures = {ex.submit(worker_sweep, job): job for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            t_nll = np.array([r["train_nll_mean"] for r in res["rows"]])
            te_nll = np.array([r["test_nll_mean"] for r in res["rows"]])
            print(
                f"  done {res['target']:<10s} ← {res['source']:<10s}  "
                f"train ∈ [{t_nll.min():.5f}..{t_nll.max():.5f}] range {t_nll.max()-t_nll.min():.5f}  "
                f"test ∈ [{te_nll.min():.5f}..{te_nll.max():.5f}] range {te_nll.max()-te_nll.min():.5f}"
            )
    print(f"\nSweep done in {time.time() - t_sweep:.1f}s wall.")

    # Order results by (target_idx, source_idx)
    results.sort(key=lambda r: (r["target_idx"], r["source_idx"]))

    # === Plot grids ===
    ch_colors = {"searches": "#2E5EAA", "to_cart": "#7B3FAA", "to_ord": "#D2691E"}

    def plot_grid(metric_key: str, out_name: str, title: str):
        fig, axes = plt.subplots(n_ch, n_ch, figsize=(5.2 * n_ch, 3.8 * n_ch), squeeze=False)
        ylabel_map = {"train_nll_mean": "train NLL / n", "test_nll_mean": "test NLL / n"}
        for r in results:
            i = r["target_idx"]; j = r["source_idx"]
            ax = axes[i][j]
            target = r["target"]; source = r["source"]
            xs = np.array(r["pin_grid"])
            ys = np.array([row[metric_key] for row in r["rows"]])
            anchor = r["anchor_pin_val"]
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=ch_colors[target])
            k_min = int(np.argmin(ys))
            ax.scatter([xs[k_min]], [ys[k_min]], s=120, facecolors="none",
                       edgecolors=ch_colors[target], linewidths=2.2, zorder=4,
                       label=f"min @ α={xs[k_min]:.3f}  (NLL={ys[k_min]:.5f})")
            ax.axvline(anchor, color="#888888", linestyle="--", linewidth=0.9,
                       label=f"unfixed anchor = {anchor:.4f}")
            rng = ys.max() - ys.min()
            pad = max(rng * 0.15, 1e-5)
            ax.set_ylim(ys.min() - pad, ys.max() + pad)
            ax.set_title(
                rf"$\alpha[\,{target} \leftarrow {source}\,]$" + f"\nrange = {rng:.5f}",
                color=ch_colors[target], fontweight="bold", fontsize=11,
            )
            if i == n_ch - 1:
                ax.set_xlabel(rf"pinned $\alpha[\,{target} \leftarrow {source}\,]$",
                              fontsize=10)
            if j == 0:
                ax.set_ylabel(ylabel_map[metric_key], fontsize=10)
            ax.grid(linestyle=":", alpha=0.5)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(OUT_DIR / out_name, dpi=140)
        plt.close(fig)

    plot_grid(
        "train_nll_mean", "train_nll_grid.png",
        r"Profile train NLL vs pinned $\alpha[i \leftarrow j]$"
        r"  (other params re-optimized at $\lambda_{\ell_2} = 1$)",
    )
    plot_grid(
        "test_nll_mean", "test_nll_grid.png",
        r"Profile test NLL vs pinned $\alpha[i \leftarrow j]$"
        r"  (other params re-optimized at $\lambda_{\ell_2} = 1$)",
    )

    # === Save CSV + JSON ===
    long_rows = []
    for res in results:
        for row in res["rows"]:
            long_rows.append({
                "target": res["target"],
                "source": res["source"],
                "anchor_pin_val": res["anchor_pin_val"],
                "pin_value": row["pin_value"],
                "alpha_full": row["alpha_full"],
                "train_nll_mean": row["train_nll_mean"],
                "test_nll_mean": row["test_nll_mean"],
                "lam_u_mean": row["lam_u_mean"],
                "mean_baseline_train": row["mean_baseline_train"],
                "mean_hawkes_train": row["mean_hawkes_train"],
                "mean_baseline_test": row["mean_baseline_test"],
                "mean_hawkes_test": row["mean_hawkes_test"],
                "converged": row["converged"],
            })
    pd.DataFrame(long_rows).to_csv(OUT_DIR / "sweep_results.csv", index=False)

    summary = {
        "channels": list(CHANNELS),
        "lambda_l2": LAMBDA_L2,
        "alpha_l2": ALPHA_L2,
        "grid_size": GRID_SIZE,
        "grid_min_upper": GRID_MIN_UPPER,
        "grid_upper_mult": GRID_UPPER_MULT,
        "cold_max_iter": COLD_MAX_ITER,
        "warm_max_iter": WARM_MAX_ITER,
        "n_polish_passes": N_POLISH_PASSES,
        "train_window": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "test_window": [str(TEST_START.date()), str(TEST_END.date())],
        "n_users": n_users,
        "anchors": [
            {
                "target": a["target"],
                "alpha": a["alpha"].tolist(),
                "lam_u_mean": float(a["lam_u"].mean()),
                "converged": a["converged"],
                "n_iter": a["n_iter"],
            }
            for a in anchors
        ],
        "sweeps": results,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()

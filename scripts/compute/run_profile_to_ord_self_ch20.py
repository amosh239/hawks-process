"""Chapter 20: Profile-likelihood sweep over α[to_ord ← to_ord].

For each value of α[to_ord ← to_ord] on the grid {0, 0.01, ..., 0.1}:
  - pin α[to_ord ← to_ord] to that value;
  - re-optimize the remaining Joint Hawkes parameters (λ_u for 10K users,
    α[to_ord ← searches], α[to_ord ← to_cart]) on the 207d train at λ_l2 = 1;
  - record train NLL and the values of the free parameters.

Output: profile NLL curve  L_profile(α[to_ord ← to_ord]) = min over the rest.
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
)


CHANNELS = ("searches", "to_cart", "to_ord")
TARGET = "to_ord"
FIXED_SOURCE = "to_ord"            # α[target ← FIXED_SOURCE] is pinned
HALF_LIVES = (1.0,)
WINDOW_SIZE = 7
ALPHA_L2 = 1e-4
LAMBDA_L2 = 1.0
MAX_ITER_COLD = 3000               # cold-start fit (first point)
MAX_ITER_WARM = 1200               # warm-start fits (subsequent)
ANCHOR_INDEX = 3                   # start sweep at PIN_GRID[3] = 0.03 (closest to unfixed optimum ≈ 0.032)
N_POLISH_PASSES = 2                # polish: re-sweep from current-best as new anchor

# Grid for the pinned α[to_ord ← to_ord]
PIN_GRID = tuple(round(v, 3) for v in np.linspace(0.0, 0.1, 11))  # 0, 0.01, ..., 0.10

TRAIN_START = pd.Timestamp("2025-01-15")
TRAIN_END = pd.Timestamp("2025-08-09")
TEST_START = pd.Timestamp("2025-08-10")
TEST_END = pd.Timestamp("2025-09-30")

OUT_DIR = Path("diploma/reports/20_profile_to_ord_self")


def fit_with_pinned_alpha(
    user_idx: np.ndarray, y: np.ndarray, b: np.ndarray, states: np.ndarray,
    n_users: int, lambda_l2: float, alpha_l2: float,
    fixed_idx: int, fixed_value: float, max_iter: int,
    lam_init: np.ndarray | None = None, alpha_free_init: np.ndarray | None = None,
) -> dict:
    """Same as fit_joint_hawkes but α[fixed_idx] is pinned to fixed_value.

    Optimizes over (λ_u of length n_users, free α of length n_alpha - 1).
    Supports warm-start via lam_init / alpha_free_init.
    """
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

    def fg(params: np.ndarray) -> tuple[float, np.ndarray]:
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
    pin_value: float, *, lam_init=None, alpha_free_init=None, max_iter: int,
    y_tr, b_tr, user_idx, n_users, states, fixed_idx,
    y_te=None, b_te=None, test_user_idx=None, states_test=None,
) -> dict:
    t0 = time.time()
    out = fit_with_pinned_alpha(
        user_idx=user_idx, y=y_tr, b=b_tr, states=states.astype(float),
        n_users=n_users, lambda_l2=LAMBDA_L2, alpha_l2=ALPHA_L2,
        fixed_idx=fixed_idx, fixed_value=float(pin_value), max_iter=max_iter,
        lam_init=lam_init, alpha_free_init=alpha_free_init,
    )
    elapsed = time.time() - t0
    lam_u = out["lam_u"]

    # Train metrics
    lam_u_t = lam_u[user_idx]
    baseline_part = lam_u_t * b_tr
    hawkes_part = states.astype(float) @ out["alpha_full"]
    lam_t = np.clip(baseline_part + hawkes_part, 1e-8, None)
    train_nll_mean = float(evaluate_count_forecast(y_tr, lam_t)["mean_poisson_nll"])
    mean_baseline = float(baseline_part.mean())
    mean_hawkes = float(hawkes_part.mean())

    record = {
        "pin_value": float(pin_value),
        "alpha_full": out["alpha_full"].tolist(),
        "lam_u": lam_u,
        "alpha_free": out["alpha_free"],
        "lam_u_mean": float(lam_u.mean()),
        "lam_u_std": float(lam_u.std()),
        "lam_u_floor_share": float((lam_u <= 0.001 + 1e-6).mean()),
        "train_loss": out["train_loss"],
        "train_nll_mean": train_nll_mean,
        "mean_baseline_train": mean_baseline,
        "mean_hawkes_train": mean_hawkes,
        "converged": out["converged"],
        "n_iter": out["n_iter"],
        "fit_time_s": elapsed,
    }

    # Test metrics (if provided): users unseen on train fall back to λ_u = 1
    if y_te is not None:
        lam_u_te = np.where(
            test_user_idx >= 0, lam_u[np.maximum(test_user_idx, 0)], 1.0
        )
        baseline_te = lam_u_te * b_te
        hawkes_te = states_test.astype(float) @ out["alpha_full"]
        lam_te = np.clip(baseline_te + hawkes_te, 1e-8, None)
        record["test_nll_mean"] = float(
            evaluate_count_forecast(y_te, lam_te)["mean_poisson_nll"]
        )
        record["mean_baseline_test"] = float(baseline_te.mean())
        record["mean_hawkes_test"] = float(hawkes_te.mean())
        record["test_users_unseen_share"] = float((test_user_idx < 0).mean())

    return record


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # Build baseline for the target channel (re-using the global rolling-seasonal model)
    y_tr = train_df[TARGET].to_numpy(dtype=float)
    y_te = test_df[TARGET].to_numpy(dtype=float)
    train_daily_mean = train_df.groupby("event_date")[TARGET].mean().sort_index()
    rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean, daily_mean_full[TARGET],
    )
    b_tr = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    b_te = rs.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)
    print(f"  target = {TARGET}, y_train mean = {y_tr.mean():.4f}, y_test mean = {y_te.mean():.4f}")
    print(f"  test rows with unseen user: {(test_user_idx < 0).sum():,} "
          f"({(test_user_idx < 0).mean() * 100:.2f}%)")

    fixed_idx = CHANNELS.index(FIXED_SOURCE)
    print(f"  pinned source = {FIXED_SOURCE} (alpha index {fixed_idx})")

    print(f"\nSequential sweep over {len(PIN_GRID)} pin values with warm-start...")
    print(f"  pin grid: {[float(v) for v in PIN_GRID]}")
    print(f"  anchor index = {ANCHOR_INDEX}  (pin = {PIN_GRID[ANCHOR_INDEX]})")
    t_global = time.time()

    common = dict(
        y_tr=y_tr, b_tr=b_tr, user_idx=train_user_idx, n_users=n_users,
        states=states_train, fixed_idx=fixed_idx,
        y_te=y_te, b_te=b_te, test_user_idx=test_user_idx, states_test=states_test,
    )

    rows_by_pin: dict[float, dict] = {}

    # 1) cold-start at anchor
    anchor_pin = float(PIN_GRID[ANCHOR_INDEX])
    res_anchor = fit_one_point(
        anchor_pin, lam_init=None, alpha_free_init=None,
        max_iter=MAX_ITER_COLD, **common,
    )
    rows_by_pin[anchor_pin] = res_anchor
    print(
        f"  [anchor] pin={anchor_pin:>5.3f}  α=[{res_anchor['alpha_full'][0]:.4f}, "
        f"{res_anchor['alpha_full'][1]:.4f}, {anchor_pin:.3f}]  "
        f"⟨λ_u⟩={res_anchor['lam_u_mean']:.4f}  "
        f"train NLL/n={res_anchor['train_nll_mean']:.6f}  test NLL/n={res_anchor['test_nll_mean']:.6f}  "
        f"({res_anchor['fit_time_s']:.1f}s, n_iter={res_anchor['n_iter']}, ok={res_anchor['converged']})"
    )

    # 2) sweep downwards from anchor (anchor-1 down to 0), warm-starting from previous
    prev = res_anchor
    for idx in range(ANCHOR_INDEX - 1, -1, -1):
        pin_v = float(PIN_GRID[idx])
        r = fit_one_point(
            pin_v, lam_init=prev["lam_u"], alpha_free_init=prev["alpha_free"],
            max_iter=MAX_ITER_WARM, **common,
        )
        rows_by_pin[pin_v] = r
        prev = r
        print(
            f"  [down ] pin={pin_v:>5.3f}  α=[{r['alpha_full'][0]:.4f}, "
            f"{r['alpha_full'][1]:.4f}, {pin_v:.3f}]  "
            f"⟨λ_u⟩={r['lam_u_mean']:.4f}  "
            f"train NLL/n={r['train_nll_mean']:.6f}  test NLL/n={r['test_nll_mean']:.6f}  "
            f"({r['fit_time_s']:.1f}s, n_iter={r['n_iter']}, ok={r['converged']})"
        )

    # 3) sweep upwards from anchor (anchor+1 up to end), warm-starting from previous
    prev = res_anchor
    for idx in range(ANCHOR_INDEX + 1, len(PIN_GRID)):
        pin_v = float(PIN_GRID[idx])
        r = fit_one_point(
            pin_v, lam_init=prev["lam_u"], alpha_free_init=prev["alpha_free"],
            max_iter=MAX_ITER_WARM, **common,
        )
        rows_by_pin[pin_v] = r
        prev = r
        print(
            f"  [up   ] pin={pin_v:>5.3f}  α=[{r['alpha_full'][0]:.4f}, "
            f"{r['alpha_full'][1]:.4f}, {pin_v:.3f}]  "
            f"⟨λ_u⟩={r['lam_u_mean']:.4f}  "
            f"train NLL/n={r['train_nll_mean']:.6f}  test NLL/n={r['test_nll_mean']:.6f}  "
            f"({r['fit_time_s']:.1f}s, n_iter={r['n_iter']}, ok={r['converged']})"
        )

    # Polish passes: re-anchor at current best and re-sweep with warm-start.
    # Keep a new point's solution only if its train_loss is lower than what we already have.
    for polish_iter in range(1, N_POLISH_PASSES + 1):
        losses = np.array([rows_by_pin[float(v)]["train_loss"] for v in PIN_GRID])
        best_idx = int(np.argmin(losses))
        print(f"\n[polish {polish_iter}] new anchor = PIN_GRID[{best_idx}] = {PIN_GRID[best_idx]} "
              f"(current best train_loss = {losses[best_idx]:.2f})")
        prev = rows_by_pin[float(PIN_GRID[best_idx])]
        for idx in range(best_idx - 1, -1, -1):
            pin_v = float(PIN_GRID[idx])
            r = fit_one_point(
                pin_v, lam_init=prev["lam_u"], alpha_free_init=prev["alpha_free"],
                max_iter=MAX_ITER_WARM, **common,
            )
            if r["train_loss"] < rows_by_pin[pin_v]["train_loss"] - 1e-6:
                old = rows_by_pin[pin_v]["train_nll_mean"]
                rows_by_pin[pin_v] = r
                marker = f"  improved: {old:.6f} -> {r['train_nll_mean']:.6f}"
            else:
                marker = "  (no improvement)"
            print(f"  [polish↓] pin={pin_v:>5.3f} train NLL/n={r['train_nll_mean']:.6f}{marker}")
            prev = r
        prev = rows_by_pin[float(PIN_GRID[best_idx])]
        for idx in range(best_idx + 1, len(PIN_GRID)):
            pin_v = float(PIN_GRID[idx])
            r = fit_one_point(
                pin_v, lam_init=prev["lam_u"], alpha_free_init=prev["alpha_free"],
                max_iter=MAX_ITER_WARM, **common,
            )
            if r["train_loss"] < rows_by_pin[pin_v]["train_loss"] - 1e-6:
                old = rows_by_pin[pin_v]["train_nll_mean"]
                rows_by_pin[pin_v] = r
                marker = f"  improved: {old:.6f} -> {r['train_nll_mean']:.6f}"
            else:
                marker = "  (no improvement)"
            print(f"  [polish↑] pin={pin_v:>5.3f} train NLL/n={r['train_nll_mean']:.6f}{marker}")
            prev = r

    print(f"\nAll done in {time.time() - t_global:.1f}s wall.")

    # Strip the heavy lam_u arrays before saving to JSON
    rows = []
    for v in PIN_GRID:
        r = rows_by_pin[float(v)]
        r_serializable = {k: r[k] for k in r if k not in ("lam_u", "alpha_free")}
        rows.append(r_serializable)

    pin_vals = np.array([r["pin_value"] for r in rows])
    train_nll = np.array([r["train_nll_mean"] for r in rows])
    test_nll = np.array([r["test_nll_mean"] for r in rows])
    lam_u_mean = np.array([r["lam_u_mean"] for r in rows])
    alpha_searches = np.array([r["alpha_full"][CHANNELS.index("searches")] for r in rows])
    alpha_to_cart = np.array([r["alpha_full"][CHANNELS.index("to_cart")] for r in rows])
    mean_baseline = np.array([r["mean_baseline_train"] for r in rows])
    mean_hawkes = np.array([r["mean_hawkes_train"] for r in rows])

    # ----- Plot 1: profile train NLL & test NLL vs pinned α (1×2 row, tight y-axes) -----
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, vals, color, title, ylabel in [
        (axes[0], train_nll, "#D2691E", "Profile train NLL",
         "profile train NLL / n"),
        (axes[1], test_nll, "#2E5EAA", "Profile test NLL",
         "profile test NLL / n"),
    ]:
        ax.plot(pin_vals, vals, marker="o", linewidth=2.0, color=color)
        i_min = int(np.argmin(vals))
        ax.scatter([pin_vals[i_min]], [vals[i_min]], s=140, facecolors="none",
                   edgecolors=color, linewidths=2.0, zorder=4,
                   label=f"min @ α={pin_vals[i_min]:g}  (NLL={vals[i_min]:.6f})")
        rng = vals.max() - vals.min()
        pad = max(rng * 0.15, 0.0001)
        ax.set_ylim(vals.min() - pad, vals.max() + pad)
        ax.set_xlabel(rf"pinned $\alpha[\,{TARGET} \leftarrow {FIXED_SOURCE}\,]$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=color, fontweight="bold")
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(
        rf"Profile NLL vs $\alpha[\,{TARGET} \leftarrow {FIXED_SOURCE}\,]$"
        rf"  (other params re-optimized at $\lambda_{{\ell_2}} = {LAMBDA_L2:g}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "profile_nll.png", dpi=150)
    plt.close(fig)

    # ----- Plot 2: companion parameters vs pinned α -----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax = axes[0]
    ax.plot(pin_vals, alpha_searches, marker="o", color="#2E5EAA", label=r"$\alpha[\,t\leftarrow searches\,]$")
    ax.plot(pin_vals, alpha_to_cart, marker="o", color="#7B3FAA", label=r"$\alpha[\,t\leftarrow to\_cart\,]$")
    ax.set_xlabel(rf"pinned $\alpha[\,t\leftarrow {FIXED_SOURCE}\,]$")
    ax.set_ylabel("re-optimized α")
    ax.set_title("Other α components")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(pin_vals, lam_u_mean, marker="o", color="#117733")
    ax.set_xlabel(rf"pinned $\alpha[\,t\leftarrow {FIXED_SOURCE}\,]$")
    ax.set_ylabel(r"$\overline{\lambda_u}$  (re-optimized)")
    ax.set_title("Mean per-user multiplier")
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.plot(pin_vals, mean_baseline, marker="o", color="#444444",
            label=r"$\overline{\lambda_u \cdot b_t}$")
    ax.plot(pin_vals, mean_hawkes, marker="o", color="#D2691E",
            label=r"$\overline{\alpha^\top z}$")
    ax.set_xlabel(rf"pinned $\alpha[\,t\leftarrow {FIXED_SOURCE}\,]$")
    ax.set_ylabel("mean per-row contribution")
    ax.set_title("Decomposition baseline vs Hawkes")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(
        rf"Re-optimized companion parameters vs pinned $\alpha[\,{TARGET}\leftarrow {FIXED_SOURCE}\,]$",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "companion_params.png", dpi=150)
    plt.close(fig)

    # ----- CSV + JSON -----
    pd.DataFrame([
        {
            "alpha_to_ord_to_ord": r["pin_value"],
            "train_nll_mean": r["train_nll_mean"],
            "alpha_searches": r["alpha_full"][CHANNELS.index("searches")],
            "alpha_to_cart": r["alpha_full"][CHANNELS.index("to_cart")],
            "lam_u_mean": r["lam_u_mean"],
            "lam_u_std": r["lam_u_std"],
            "lam_u_floor_share": r["lam_u_floor_share"],
            "test_nll_mean": r["test_nll_mean"],
            "mean_baseline_train": r["mean_baseline_train"],
            "mean_hawkes_train": r["mean_hawkes_train"],
            "mean_baseline_test": r["mean_baseline_test"],
            "mean_hawkes_test": r["mean_hawkes_test"],
            "train_loss_total": r["train_loss"],
            "converged": r["converged"],
            "n_iter": r["n_iter"],
        }
        for r in rows
    ]).to_csv(OUT_DIR / "profile_sweep.csv", index=False)

    i_min_train = int(np.argmin(train_nll))
    i_min_test = int(np.argmin(test_nll))
    summary = {
        "target": TARGET,
        "fixed_source": FIXED_SOURCE,
        "fixed_idx": fixed_idx,
        "pin_grid": list(PIN_GRID),
        "lambda_l2": LAMBDA_L2,
        "alpha_l2": ALPHA_L2,
        "max_iter_cold": MAX_ITER_COLD,
        "max_iter_warm": MAX_ITER_WARM,
        "anchor_index": ANCHOR_INDEX,
        "train_window": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "test_window": [str(TEST_START.date()), str(TEST_END.date())],
        "n_users": n_users,
        "rows": rows,
        "argmin_pin_train": float(pin_vals[i_min_train]),
        "min_train_nll": float(train_nll[i_min_train]),
        "argmin_pin_test": float(pin_vals[i_min_test]),
        "min_test_nll": float(test_nll[i_min_test]),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save lam_u vectors for downstream analysis
    np.savez_compressed(
        OUT_DIR / "lam_u_per_pin.npz",
        pin_grid=np.array(PIN_GRID, dtype=float),
        lam_u=np.stack([rows_by_pin[float(v)]["lam_u"].astype(np.float32) for v in PIN_GRID]),
    )

    print(f"\nDone. Saved artifacts to {OUT_DIR}")
    print(f"\nargmin (train) pinned α = {pin_vals[i_min_train]:g},  min train NLL/n = {train_nll[i_min_train]:.6f}")
    print(f"argmin (test)  pinned α = {pin_vals[i_min_test]:g},  min test NLL/n  = {test_nll[i_min_test]:.6f}")


if __name__ == "__main__":
    main()

"""Chapter 18: cross-channel Joint Hawkes WITHOUT λ_u regularization.

Same protocol as chapter 17 (10 random 100d windows, 66d train + 34d test,
3 channels, hl=1d) but `lambda_l2 = 0.0` — no L2 penalty on (λ_u − 1)²
at all. Per-user multiplier becomes raw MLE (subject only to the
`λ_u ∈ [0.001, 50]` box constraint).

Goal: separate the effect of "model class" (per-user multiplier free)
from the effect of "shrinkage prior". With the L2-toward-1 prior gone,
Joint should fit train strictly better than Scaled (it's a strict
superset of `c · μ · b_t` — c is just a single shared multiplier, while
λ_u is per-user).

Artifacts under `diploma/reports/18_joint_unregularized/`.
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

from src.diploma_baselines.data import load_daily_grid
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_baselines.models import (
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_user_states_cache,
    fit_joint_hawkes,
)


CHANNELS = ("searches", "to_cart", "to_ord")
HALF_LIVES = (1.0,)
WINDOW_LEN = 100
N_WINDOWS = 10
TRAIN_LEN = (WINDOW_LEN // 3) * 2  # = 66
WINDOW_SIZE = 7
LAMBDA_L2 = 0.0   # <-- the only meaningful difference vs ch.17
ALPHA_L2 = 1e-4
MAX_ITER = 500    # bumped from 300 since unregularized fit needs more iterations

GLOBAL_START = pd.Timestamp("2025-01-15")
GLOBAL_END = pd.Timestamp("2025-10-31")
SEED = 42

OUT_DIR = Path("diploma/reports/18_joint_unregularized")


def sample_starts(n_days: int, m: int, seed: int) -> list[pd.Timestamp]:
    earliest = GLOBAL_START
    latest = GLOBAL_END - pd.Timedelta(days=n_days - 1)
    n_possible = (latest - earliest).days + 1
    rng = np.random.default_rng(seed)
    if n_possible <= m:
        offsets = list(range(n_possible))
    else:
        offsets = sorted(rng.choice(n_possible, size=m, replace=False).tolist())
    return [earliest + pd.Timedelta(days=int(o)) for o in offsets]


def fit_one_channel_joint_unreg(
    target: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    daily_mean_full_for_target: pd.Series,
    states_train: np.ndarray,
    states_test: np.ndarray,
):
    """Joint Hawkes fit with `lambda_l2 = 0` (no shrinkage on λ_u)."""
    y_train = train_df[target].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    train_daily_mean = train_df.groupby("event_date")[target].mean().sort_index()
    rs = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
        train_daily_mean, daily_mean_full_for_target,
    )
    base_train = rs.predict_for_dates(train_df["event_date"]).to_numpy(dtype=float)
    base_test = rs.predict_for_dates(test_df["event_date"]).to_numpy(dtype=float)

    train_uids = train_df["user_id"].to_numpy()
    test_uids = test_df["user_id"].to_numpy()
    scaler = PersonalizedGammaPoissonScaler().fit(train_uids, y_train, base_train)
    pers_train = scaler.predict(train_uids, base_train, method="posterior_mean")
    pers_test = scaler.predict(test_uids, base_test, method="posterior_mean")
    personalized_train_nll = float(evaluate_count_forecast(y_train, pers_train)["mean_poisson_nll"])
    personalized_test_nll = float(evaluate_count_forecast(y_test, pers_test)["mean_poisson_nll"])

    unique_train_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
    n_users = int(len(unique_train_uids))

    j = fit_joint_hawkes(
        user_idx=train_user_idx,
        y=y_train,
        b=base_train,
        states=states_train.astype(float),
        n_users=n_users,
        lambda_l2=LAMBDA_L2,   # 0 — no shrinkage
        alpha_l2=ALPHA_L2,
        max_iter=MAX_ITER,
    )
    lam_u = j.lam_u
    alpha = j.alpha

    # Train metrics
    lam_u_for_train = lam_u[train_user_idx]
    lam_train = np.clip(
        lam_u_for_train * base_train + states_train.astype(float) @ alpha, 1e-8, None,
    )
    hawkes_train_nll = float(evaluate_count_forecast(y_train, lam_train)["mean_poisson_nll"])

    # Test metrics: λ_u = 1 for unseen users (same as ch.17)
    uid_to_idx = {int(u): i for i, u in enumerate(unique_train_uids)}
    test_user_idx = np.array(
        [uid_to_idx.get(int(u), -1) for u in test_uids], dtype=np.int64
    )
    lam_u_for_test = np.where(test_user_idx >= 0, lam_u[np.maximum(test_user_idx, 0)], 1.0)
    lam_test = np.clip(
        lam_u_for_test * base_test + states_test.astype(float) @ alpha, 1e-8, None,
    )
    hawkes_test_nll = float(evaluate_count_forecast(y_test, lam_test)["mean_poisson_nll"])

    # How many λ_u sit at the lower box boundary? — diagnostic for "all-zero users"
    floor_share = float((lam_u <= 0.001 + 1e-6).mean())

    return {
        "target": target,
        "alpha": np.asarray(alpha, dtype=float),
        "alpha_norm": float(np.linalg.norm(alpha)),
        "lam_u_mean": float(lam_u.mean()),
        "lam_u_median": float(np.median(lam_u)),
        "lam_u_std": float(lam_u.std(ddof=1)),
        "lam_u_q10": float(np.quantile(lam_u, 0.1)),
        "lam_u_q90": float(np.quantile(lam_u, 0.9)),
        "lam_u_floor_share": floor_share,
        "personalized_train_nll": personalized_train_nll,
        "personalized_test_nll": personalized_test_nll,
        "hawkes_train_nll": hawkes_train_nll,
        "hawkes_test_nll": hawkes_test_nll,
        "delta_nll_train": hawkes_train_nll - personalized_train_nll,
        "delta_nll_test": hawkes_test_nll - personalized_test_nll,
        "converged": bool(j.converged),
        "n_iter": int(j.n_iter),
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

    print("Building Hawkes states cache once...")
    cache = build_user_states_cache(full_df, features=CHANNELS, half_lives=HALF_LIVES)

    starts = sample_starts(WINDOW_LEN, N_WINDOWS, seed=SEED)
    print(f"Sampled {len(starts)} starts: {[str(s.date()) for s in starts]}")

    all_lam_u_mean = np.zeros((N_WINDOWS, n_ch))
    all_lam_u_floor_share = np.zeros((N_WINDOWS, n_ch))
    all_alpha = np.zeros((N_WINDOWS, n_ch, n_ch))
    all_personalized_train_nll = np.zeros((N_WINDOWS, n_ch))
    all_hawkes_train_nll = np.zeros((N_WINDOWS, n_ch))
    all_personalized_nll = np.zeros((N_WINDOWS, n_ch))
    all_hawkes_nll = np.zeros((N_WINDOWS, n_ch))
    per_window_meta: list[dict] = []
    long_rows: list[dict] = []

    for w, start in enumerate(starts):
        window_end = start + pd.Timedelta(days=WINDOW_LEN - 1)
        train_end = start + pd.Timedelta(days=TRAIN_LEN - 1)
        test_start = train_end + pd.Timedelta(days=1)
        print(
            f"\n=== window {w + 1}/{N_WINDOWS}: "
            f"{start.date()} .. {window_end.date()} (train .. {train_end.date()}) ==="
        )

        block_df = full_df.loc[
            (full_df["event_date"] >= start) & (full_df["event_date"] <= window_end)
        ].copy()
        train_df = block_df[block_df["event_date"] <= train_end].copy()
        test_df = block_df[block_df["event_date"] >= test_start].copy()
        states_train = cache.gather_for(train_df)
        states_test = cache.gather_for(test_df)

        per_window_meta.append(
            {
                "window_idx": w,
                "start": str(start.date()),
                "end": str(window_end.date()),
                "train_end": str(train_end.date()),
                "n_train_rows": int(len(train_df)),
                "n_test_rows": int(len(test_df)),
            }
        )

        for i, target in enumerate(CHANNELS):
            t0 = time.time()
            res = fit_one_channel_joint_unreg(
                target=target,
                train_df=train_df,
                test_df=test_df,
                daily_mean_full_for_target=daily_mean_full[target],
                states_train=states_train,
                states_test=states_test,
            )
            elapsed = time.time() - t0
            all_lam_u_mean[w, i] = res["lam_u_mean"]
            all_lam_u_floor_share[w, i] = res["lam_u_floor_share"]
            all_alpha[w, i, :] = res["alpha"]
            all_personalized_train_nll[w, i] = res["personalized_train_nll"]
            all_hawkes_train_nll[w, i] = res["hawkes_train_nll"]
            all_personalized_nll[w, i] = res["personalized_test_nll"]
            all_hawkes_nll[w, i] = res["hawkes_test_nll"]
            print(
                f"  {target:<10s} mean λ_u={res['lam_u_mean']:.4f} "
                f"(floor={res['lam_u_floor_share']*100:5.1f}%)  "
                f"||α||={res['alpha_norm']:.4f}  "
                f"Δ train={res['delta_nll_train']:+.4f}  Δ test={res['delta_nll_test']:+.4f}  "
                f"({elapsed:.1f}s, n_iter={res['n_iter']}, ok={res['converged']})"
            )
            for j, src in enumerate(CHANNELS):
                long_rows.append(
                    {
                        "window_idx": w,
                        "start": str(start.date()),
                        "target": target,
                        "source": src,
                        "alpha": float(res["alpha"][j]),
                        "lam_u_mean": float(res["lam_u_mean"]),
                        "lam_u_floor_share": float(res["lam_u_floor_share"]),
                    }
                )

    # === Aggregates ===
    alpha_mean = all_alpha.mean(axis=0)
    alpha_std = all_alpha.std(axis=0, ddof=1)
    alpha_q10 = np.quantile(all_alpha, 0.1, axis=0)
    alpha_q90 = np.quantile(all_alpha, 0.9, axis=0)
    alpha_min = all_alpha.min(axis=0)
    alpha_max = all_alpha.max(axis=0)
    alpha_zero_share = (all_alpha == 0).sum(axis=0) / N_WINDOWS

    lam_u_mean_mean = all_lam_u_mean.mean(axis=0)
    lam_u_mean_std = all_lam_u_mean.std(axis=0, ddof=1)
    lam_u_floor_mean = all_lam_u_floor_share.mean(axis=0)

    pd.DataFrame(long_rows).to_csv(OUT_DIR / "alpha_per_window.csv", index=False)
    pd.DataFrame(alpha_mean, index=list(CHANNELS), columns=list(CHANNELS)).to_csv(
        OUT_DIR / "alpha_mean.csv"
    )
    pd.DataFrame(alpha_std, index=list(CHANNELS), columns=list(CHANNELS)).to_csv(
        OUT_DIR / "alpha_std.csv"
    )

    # === Heatmap of mean(α) with mean ± std ===
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    vmax = float(np.max(np.abs(alpha_mean))) if np.any(np.abs(alpha_mean) > 0) else 0.01
    im = ax.imshow(alpha_mean, cmap="RdBu_r", vmin=-vmax, vmax=+vmax, aspect="auto")
    ax.set_xticks(range(n_ch)); ax.set_xticklabels(CHANNELS, rotation=20)
    ax.set_yticks(range(n_ch)); ax.set_yticklabels(CHANNELS)
    ax.set_xlabel("Source channel  (j)")
    ax.set_ylabel("Target channel  (i)")
    ax.set_title(
        r"Joint Hawkes (no $\lambda_u$ shrinkage) $\alpha_{i,j}$: mean $\pm$ std over 10 random 100d windows"
    )
    for i in range(n_ch):
        for j in range(n_ch):
            m = alpha_mean[i, j]; s = alpha_std[i, j]
            color = "white" if abs(m) > 0.5 * vmax else "black"
            ax.text(j, i, f"{m:.3f}\n±{s:.3f}",
                    ha="center", va="center", fontsize=11, color=color)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"mean $\alpha_{i,j}$")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_heatmap_with_ci.png", dpi=150)
    plt.close(fig)

    # === Three-way side-by-side compare: Scaled (ch.16) / Joint L2=1 (ch.17) / Joint L2=0 (ch.18) ===
    ch16_path = Path("diploma/reports/16_cross_channel_bootstrap/summary.json")
    ch17_path = Path("diploma/reports/17_cross_channel_joint_bootstrap/summary.json")
    if ch16_path.exists() and ch17_path.exists():
        ch16 = json.loads(ch16_path.read_text())
        ch17 = json.loads(ch17_path.read_text())
        alpha_16 = np.array([[ch16["alpha_mean"][t][s] for s in CHANNELS] for t in CHANNELS])
        alpha_17 = np.array([[ch17["alpha_mean"][t][s] for s in CHANNELS] for t in CHANNELS])
        global_vmax = float(max(np.max(np.abs(alpha_16)), np.max(np.abs(alpha_17)), np.max(np.abs(alpha_mean))))

        fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4))
        for ax_idx, (mat, title, color) in enumerate([
            (alpha_16, "Ch.16 — Scaled-baseline\n(c · μ_u^EB · b_t + α^T s)", "#2E5EAA"),
            (alpha_17, "Ch.17 — Joint, λ_l2 = 1.0\n(λ_u · b_t + α^T s, soft shrink to 1)", "#7B3FAA"),
            (alpha_mean, "Ch.18 — Joint, λ_l2 = 0\n(λ_u free, no shrinkage)", "#D2691E"),
        ]):
            ax = axes[ax_idx]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-global_vmax, vmax=+global_vmax, aspect="auto")
            ax.set_xticks(range(n_ch)); ax.set_xticklabels(CHANNELS, rotation=20)
            ax.set_yticks(range(n_ch)); ax.set_yticklabels(CHANNELS)
            ax.set_title(title, color=color, fontweight="bold", fontsize=10)
            for i in range(n_ch):
                for j in range(n_ch):
                    m = mat[i, j]
                    color_t = "white" if abs(m) > 0.5 * global_vmax else "black"
                    ax.text(j, i, f"{m:.3f}", ha="center", va="center", fontsize=11, color=color_t)
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle(
            r"mean($\alpha_{i,j}$) across same 10 random 100d windows: Scaled / Joint(L2=1) / Joint(L2=0)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / "alpha_compare_three_way.png", dpi=150)
        plt.close(fig)

    # === 3×3 distribution grid ===
    fig, axes = plt.subplots(n_ch, n_ch, figsize=(3.0 * n_ch, 2.6 * n_ch), squeeze=False)
    rng = np.random.default_rng(0)
    for i in range(n_ch):
        for j in range(n_ch):
            ax = axes[i][j]
            vals = all_alpha[:, i, j]
            x_jit = (rng.random(len(vals)) - 0.5) * 0.4
            ax.scatter(x_jit, vals, color="#D2691E", s=40, alpha=0.7, edgecolors="white")
            ax.hlines(vals.mean(), -0.3, 0.3, color="#7A3F0F", linewidth=2.4, label="mean")
            ax.hlines([alpha_q10[i, j], alpha_q90[i, j]], -0.3, 0.3,
                      color="#0B3C5D", linewidth=1.0, linestyles="--", label="q10..q90")
            ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
            on_diag = i == j
            tt = f"{CHANNELS[i]} ← {CHANNELS[j]}" + ("  (self)" if on_diag else "")
            ax.set_title(tt, fontsize=9)
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            zero_count = int((vals == 0).sum())
            txt = f"μ={vals.mean():.4f}\nσ={vals.std(ddof=1):.4f}"
            if zero_count > 0:
                txt += f"\n#zero={zero_count}/{N_WINDOWS}"
            ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=8, color="#666",
                    ha="left", va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    fig.suptitle(
        r"Joint Hawkes ($\lambda_l2 = 0$) $\alpha_{i,j}$ distribution across 10 random 100d windows",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_distributions.png", dpi=150)
    plt.close(fig)

    # === λ_u summary ===
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    rng2 = np.random.default_rng(1)
    for i, ch in enumerate(CHANNELS):
        vals = all_lam_u_mean[:, i]
        x_jit = i + (rng2.random(len(vals)) - 0.5) * 0.32
        ax.scatter(x_jit, vals, color="#D2691E", s=44, alpha=0.7, edgecolors="white",
                   label=("per-window mean(λ_u)" if i == 0 else None))
        ax.hlines(vals.mean(), i - 0.28, i + 0.28, color="#7A3F0F",
                  linewidth=2.4, label=("aggregate mean" if i == 0 else None))
    ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle="--", label="λ_u = 1")
    ax.set_xticks(range(n_ch)); ax.set_xticklabels(CHANNELS)
    ax.set_ylabel(r"$\overline{\lambda_u}$ over train users")
    ax.set_title(rf"Joint Hawkes ($\lambda_l2 = 0$): mean(λ_u) per channel across {N_WINDOWS} windows")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lambda_u_distribution.png", dpi=150)
    plt.close(fig)

    # === Summary JSON ===
    def m_(arr: np.ndarray) -> dict[str, dict[str, float]]:
        return {t: dict(zip(CHANNELS, arr[i].tolist())) for i, t in enumerate(CHANNELS)}

    summary = {
        "window_len": WINDOW_LEN,
        "n_windows": N_WINDOWS,
        "train_len": TRAIN_LEN,
        "test_len": WINDOW_LEN - TRAIN_LEN,
        "channels": list(CHANNELS),
        "half_lives": list(HALF_LIVES),
        "lambda_l2": LAMBDA_L2,
        "alpha_l2": ALPHA_L2,
        "max_iter": MAX_ITER,
        "seed": SEED,
        "model": "joint_hawkes_unregularized",
        "windows": per_window_meta,
        "alpha_mean": m_(alpha_mean),
        "alpha_std": m_(alpha_std),
        "alpha_q10": m_(alpha_q10),
        "alpha_q90": m_(alpha_q90),
        "alpha_min": m_(alpha_min),
        "alpha_max": m_(alpha_max),
        "alpha_zero_share": m_(alpha_zero_share),
        "lam_u_mean_mean": dict(zip(CHANNELS, lam_u_mean_mean.tolist())),
        "lam_u_mean_std": dict(zip(CHANNELS, lam_u_mean_std.tolist())),
        "lam_u_floor_share_mean": dict(zip(CHANNELS, lam_u_floor_mean.tolist())),
        "personalized_train_nll_per_window": all_personalized_train_nll.tolist(),
        "hawkes_train_nll_per_window": all_hawkes_train_nll.tolist(),
        "personalized_test_nll_per_window": all_personalized_nll.tolist(),
        "hawkes_test_nll_per_window": all_hawkes_nll.tolist(),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()

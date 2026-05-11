"""Chapter 16: cross-channel Hawkes on 10 random 100-day windows.

Same per-target Scaled-baseline Hawkes fit as chapter 15 (3 channels —
`searches`, `to_cart`, `to_ord`, single 1-day half-life), but repeated on
10 random 100-day windows sampled inside `[2025-01-15, 2025-10-31]`. Each
window is split 2/3 train (66d) / 1/3 test (34d) — same protocol as
chapter 11. From the resulting 10 fits we collect per-cell distributions
of `α_{i,j}` and `c_i` and report mean / std / quantiles as a confidence
band on the structural matrix.

Artifacts under `diploma/reports/16_cross_channel_bootstrap/`:
  alpha_per_window.csv     — 10 × 9 records of α (long format)
  alpha_mean.csv / alpha_std.csv — 3×3 matrices
  alpha_heatmap_with_ci.png — heatmap of mean(α) with mean±std annotation
  alpha_distributions.png  — 3×3 grid of strip plots, one per (target, source)
  c_distribution.png       — strip plot of c per channel
  summary.json
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
from src.diploma_baselines.models import build_user_states_cache
from scripts.compute.run_cross_channel_hawkes_ch15 import fit_one_channel


CHANNELS = ("searches", "to_cart", "to_ord")
HALF_LIVES = (1.0,)
WINDOW_LEN = 100
N_WINDOWS = 10
TRAIN_LEN = (WINDOW_LEN // 3) * 2  # = 66

GLOBAL_START = pd.Timestamp("2025-01-15")
GLOBAL_END = pd.Timestamp("2025-10-31")
SEED = 42

OUT_DIR = Path("diploma/reports/16_cross_channel_bootstrap")


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

    all_c = np.zeros((N_WINDOWS, n_ch))
    all_alpha = np.zeros((N_WINDOWS, n_ch, n_ch))
    all_baseline_train_nll = np.zeros((N_WINDOWS, n_ch))
    all_hawkes_train_nll = np.zeros((N_WINDOWS, n_ch))
    all_baseline_nll = np.zeros((N_WINDOWS, n_ch))
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
            res = fit_one_channel(
                target=target,
                channels=CHANNELS,
                full_df=full_df,
                train_df=train_df,
                test_df=test_df,
                daily_mean_full_per_channel=daily_mean_full,
                states_train=states_train,
                states_test=states_test,
            )
            elapsed = time.time() - t0
            all_c[w, i] = res["c"]
            all_alpha[w, i, :] = res["alpha"]
            all_baseline_train_nll[w, i] = res["baseline_train_nll"]
            all_hawkes_train_nll[w, i] = res["hawkes_train_nll"]
            all_baseline_nll[w, i] = res["baseline_test_nll"]
            all_hawkes_nll[w, i] = res["hawkes_test_nll"]
            print(
                f"  {target:<10s} c={res['c']:.4f} ||α||={res['alpha_norm']:.4f} "
                f"Δ train={res['delta_nll_train']:+.4f}  Δ test={res['delta_nll_test']:+.4f}  ({elapsed:.1f}s)"
            )
            for j, src in enumerate(CHANNELS):
                long_rows.append(
                    {
                        "window_idx": w,
                        "start": str(start.date()),
                        "target": target,
                        "source": src,
                        "alpha": float(res["alpha"][j]),
                        "c": float(res["c"]),
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

    c_mean = all_c.mean(axis=0)
    c_std = all_c.std(axis=0, ddof=1)

    # === Persist CSVs ===
    pd.DataFrame(long_rows).to_csv(OUT_DIR / "alpha_per_window.csv", index=False)
    pd.DataFrame(alpha_mean, index=list(CHANNELS), columns=list(CHANNELS)).to_csv(
        OUT_DIR / "alpha_mean.csv"
    )
    pd.DataFrame(alpha_std, index=list(CHANNELS), columns=list(CHANNELS)).to_csv(
        OUT_DIR / "alpha_std.csv"
    )

    # === Heatmap of mean(α) with mean±std annotation ===
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    vmax = float(np.max(np.abs(alpha_mean)))
    im = ax.imshow(alpha_mean, cmap="RdBu_r", vmin=-vmax, vmax=+vmax, aspect="auto")
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(CHANNELS, rotation=20)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(CHANNELS)
    ax.set_xlabel("Source channel  (j)")
    ax.set_ylabel("Target channel  (i)")
    ax.set_title(
        r"Cross-channel $\alpha_{i,j}$: mean $\pm$ std over 10 random 100d windows"
    )
    for i in range(n_ch):
        for j in range(n_ch):
            m = alpha_mean[i, j]
            s = alpha_std[i, j]
            color = "white" if abs(m) > 0.5 * vmax else "black"
            ax.text(
                j, i, f"{m:.3f}\n±{s:.3f}",
                ha="center", va="center", fontsize=11, color=color,
            )
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"mean $\alpha_{i,j}$")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_heatmap_with_ci.png", dpi=150)
    plt.close(fig)

    # === 3x3 grid of distributions ===
    fig, axes = plt.subplots(n_ch, n_ch, figsize=(3.0 * n_ch, 2.6 * n_ch), squeeze=False)
    rng = np.random.default_rng(0)
    for i in range(n_ch):
        for j in range(n_ch):
            ax = axes[i][j]
            vals = all_alpha[:, i, j]
            x_jit = (rng.random(len(vals)) - 0.5) * 0.4
            ax.scatter(x_jit, vals, color="#2E5EAA", s=40, alpha=0.7, edgecolors="white")
            ax.hlines(vals.mean(), -0.3, 0.3, color="#0B3C5D", linewidth=2.4, label="mean")
            ax.hlines([alpha_q10[i, j], alpha_q90[i, j]], -0.3, 0.3,
                      color="#D2691E", linewidth=1.0, linestyles="--",
                      label="q10..q90")
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            on_diag = i == j
            tt = f"{CHANNELS[i]} ← {CHANNELS[j]}" + ("  (self)" if on_diag else "")
            ax.set_title(tt, fontsize=9)
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            zero_count = int((vals == 0).sum())
            txt = f"μ={vals.mean():.4f}\nσ={vals.std(ddof=1):.4f}"
            if zero_count > 0:
                txt += f"\n#zero={zero_count}/{N_WINDOWS}"
            ax.text(
                0.05, 0.95, txt, transform=ax.transAxes, fontsize=8,
                color="#666", ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
            )
    fig.suptitle(
        r"Distribution of $\alpha_{i,j}$ across 10 random 100d windows  (66d train each)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_distributions.png", dpi=150)
    plt.close(fig)

    # === c distribution ===
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    rng2 = np.random.default_rng(1)
    for i, ch in enumerate(CHANNELS):
        vals = all_c[:, i]
        x_jit = i + (rng2.random(len(vals)) - 0.5) * 0.32
        ax.scatter(
            x_jit, vals, color="#2E5EAA", s=44, alpha=0.7, edgecolors="white",
            label=("c per window" if i == 0 else None),
        )
        ax.hlines(vals.mean(), i - 0.28, i + 0.28, color="#0B3C5D",
                  linewidth=2.4, label=("mean" if i == 0 else None))
    ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle="--", label="c = 1")
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel("c (per-channel baseline rescale)")
    ax.set_title(f"Per-channel c across {N_WINDOWS} random {WINDOW_LEN}d windows")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "c_distribution.png", dpi=150)
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
        "seed": SEED,
        "windows": per_window_meta,
        "alpha_mean": m_(alpha_mean),
        "alpha_std": m_(alpha_std),
        "alpha_q10": m_(alpha_q10),
        "alpha_q90": m_(alpha_q90),
        "alpha_min": m_(alpha_min),
        "alpha_max": m_(alpha_max),
        "alpha_zero_share": m_(alpha_zero_share),
        "c_mean": dict(zip(CHANNELS, c_mean.tolist())),
        "c_std": dict(zip(CHANNELS, c_std.tolist())),
        "baseline_train_nll_per_window": all_baseline_train_nll.tolist(),
        "hawkes_train_nll_per_window": all_hawkes_train_nll.tolist(),
        "baseline_test_nll_per_window": all_baseline_nll.tolist(),
        "hawkes_test_nll_per_window": all_hawkes_nll.tolist(),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Plot train-length scan results from saved CSV.

Reads diploma/reports/11_train_length_scan/scan_results.csv and produces:
  - mean_nll_vs_n.png:   mean NLL per model as a function of n_days
  - strip_per_n.png:     grid of strip plots, one per n value
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("diploma/reports/11_train_length_scan")

MODEL_LABELS = [
    "Global Poisson",
    "Rolling Poisson",
    "Rolling Seasonal",
    "Personalized Gamma-Poisson",
    "Scaled-baseline Hawkes",
    "Joint Hawkes (λ_u + α)",
    "Pooled Hawkes (c·b_t + α^T s)",
]

# Distinct colors per model
MODEL_COLORS = {
    "Global Poisson": "#9CA3AF",
    "Rolling Poisson": "#6B7280",
    "Rolling Seasonal": "#374151",
    "Personalized Gamma-Poisson": "#2E5EAA",
    "Scaled-baseline Hawkes": "#0B3C5D",
    "Joint Hawkes (λ_u + α)": "#7B3FAA",
    "Pooled Hawkes (c·b_t + α^T s)": "#1F8FFF",
}


def main():
    df = pd.read_csv(OUT_DIR / "scan_results.csv")
    print(f"Loaded {len(df)} runs across {df['n_days'].nunique()} n values")

    # === Plot 1: mean NLL vs n, per model ===
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    n_grid = sorted(df["n_days"].unique())
    for label in MODEL_LABELS:
        if label not in df.columns:
            continue
        means = []
        for n in n_grid:
            sub = df[df["n_days"] == n][label].dropna().to_numpy(dtype=float)
            means.append(np.mean(sub) if len(sub) else np.nan)
        color = MODEL_COLORS.get(label, "#000000")
        ax.plot(n_grid, means, marker="o", linewidth=1.8, color=color, label=label)
    ax.set_xlabel("n (длина окна, дней)")
    ax.set_ylabel("Mean test NLL per user-day (lower is better)")
    ax.set_title("Train-length scan: mean test NLL по моделям от длины окна")
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mean_nll_vs_n.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'mean_nll_vs_n.png'}")

    # === Plot 2: grid of strip plots, one per n value ===
    n_panels = len(n_grid)
    n_cols = 5
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows), squeeze=False)
    rng = np.random.default_rng(0)
    label_short = {
        "Global Poisson": "Global",
        "Rolling Poisson": "Rolling",
        "Rolling Seasonal": "RS",
        "Personalized Gamma-Poisson": "Pers GP",
        "Scaled-baseline Hawkes": "Scaled H",
        "Joint Hawkes (λ_u + α)": "Joint H",
        "Pooled Hawkes (c·b_t + α^T s)": "Pooled H",
    }

    for panel_idx, n_days in enumerate(n_grid):
        ax = axes[panel_idx // n_cols][panel_idx % n_cols]
        sub_df = df[df["n_days"] == n_days]
        m_runs = len(sub_df)
        for i, label in enumerate(MODEL_LABELS):
            if label not in df.columns:
                continue
            vals = sub_df[label].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            x_jitter = i + (rng.random(len(vals)) - 0.5) * 0.32
            ax.scatter(x_jitter, vals, s=18, alpha=0.55, color=MODEL_COLORS.get(label, "#000"),
                       edgecolors="white", linewidths=0.4)
            mean_val = float(np.mean(vals))
            ax.hlines(mean_val, i - 0.28, i + 0.28, color="#0B3C5D", linewidth=1.8, zorder=4)
        ax.set_xticks(range(len(MODEL_LABELS)))
        ax.set_xticklabels([label_short[lbl] for lbl in MODEL_LABELS], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"n = {n_days}d  ({m_runs} runs)", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if panel_idx % n_cols == 0:
            ax.set_ylabel("test NLL", fontsize=9)

    # Hide empty axes
    for k in range(n_panels, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle("Train-length scan: per-run test NLL по моделям (одна панель на n)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "strip_per_n.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'strip_per_n.png'}")

    # Save aggregated stats per (n, model)
    agg_rows = []
    for n in n_grid:
        for label in MODEL_LABELS:
            if label not in df.columns:
                continue
            vals = df[df["n_days"] == n][label].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            agg_rows.append({
                "n_days": int(n),
                "model": label,
                "n_runs": int(len(vals)),
                "mean_nll": float(np.mean(vals)),
                "median_nll": float(np.median(vals)),
                "std_nll": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min_nll": float(np.min(vals)),
                "max_nll": float(np.max(vals)),
            })
    pd.DataFrame(agg_rows).to_csv(OUT_DIR / "scan_aggregated.csv", index=False)
    print(f"Saved {OUT_DIR / 'scan_aggregated.csv'}")

    # === Plot 3: ‖α‖₂ vs n for the 3 Hawkes models ===
    hawkes_models = [
        ("Scaled-baseline Hawkes", "Scaled-baseline Hawkes_alpha_norm"),
        ("Joint Hawkes (λ_u + α)", "Joint Hawkes (λ_u + α)_alpha_norm"),
        ("Pooled Hawkes (c·b_t + α^T s)", "Pooled Hawkes (c·b_t + α^T s)_alpha_norm"),
    ]
    if all(col in df.columns for _, col in hawkes_models):
        fig, ax = plt.subplots(figsize=(10.0, 5.4))
        for label, col in hawkes_models:
            means = []; stds = []
            for n in n_grid:
                vals = df[df["n_days"] == n][col].dropna().to_numpy(dtype=float)
                means.append(np.mean(vals) if len(vals) else np.nan)
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            color = MODEL_COLORS.get(label, "#000")
            means_arr = np.array(means); stds_arr = np.array(stds)
            ax.plot(n_grid, means_arr, marker="o", linewidth=1.8, color=color, label=label)
            ax.fill_between(n_grid, means_arr - stds_arr, means_arr + stds_arr,
                            alpha=0.15, color=color, linewidth=0)
        ax.axhline(0, color="#888888", linewidth=0.6)
        ax.set_xlabel("n (длина окна, дней)")
        ax.set_ylabel("‖α‖₂ (mean ± std по подвыборкам)")
        ax.set_title("Train-length scan: норма Hawkes-коэффициентов в зависимости от длины окна")
        ax.grid(linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "alpha_norm_vs_n.png", dpi=150)
        plt.close(fig)
        print(f"Saved {OUT_DIR / 'alpha_norm_vs_n.png'}")

    # === Plot 4: mean per-user multiplier m_u vs n for the 3 Hawkes models ===
    multiplier_models = [
        ("Scaled-baseline Hawkes", "Scaled-baseline Hawkes_m_u_mean"),
        ("Joint Hawkes (λ_u + α)", "Joint Hawkes (λ_u + α)_m_u_mean"),
        ("Pooled Hawkes (c·b_t + α^T s)", "Pooled Hawkes (c·b_t + α^T s)_m_u_mean"),
    ]
    if all(col in df.columns for _, col in multiplier_models):
        fig, ax = plt.subplots(figsize=(10.0, 5.4))
        for label, col in multiplier_models:
            means = []; stds = []
            for n in n_grid:
                vals = df[df["n_days"] == n][col].dropna().to_numpy(dtype=float)
                means.append(np.mean(vals) if len(vals) else np.nan)
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            color = MODEL_COLORS.get(label, "#000")
            means_arr = np.array(means); stds_arr = np.array(stds)
            ax.plot(n_grid, means_arr, marker="o", linewidth=1.8, color=color, label=label)
            ax.fill_between(n_grid, means_arr - stds_arr, means_arr + stds_arr,
                            alpha=0.15, color=color, linewidth=0)
        ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--", label="m_u = 1")
        ax.set_xlabel("n (длина окна, дней)")
        ax.set_ylabel("mean m_u — средний множитель перед b_t")
        ax.set_title("Train-length scan: средний per-user множитель перед b_t в зависимости от длины окна")
        ax.grid(linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "m_u_mean_vs_n.png", dpi=150)
        plt.close(fig)
        print(f"Saved {OUT_DIR / 'm_u_mean_vs_n.png'}")


if __name__ == "__main__":
    main()

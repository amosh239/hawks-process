"""Slide-specific replots without Pooled Hawkes column.

Three output PNGs (re-rendered from the same CSVs as the diploma versions, just
with the Pooled Hawkes column dropped). The diploma chapters keep their
original images intact; only the presentation deck uses these slim versions.

Outputs:
  diploma/reports/blockwise_cv/cv_strip_plot_no_pooled.png
  diploma/reports/11_train_length_scan/mean_nll_vs_n_no_pooled.png
  diploma/reports/11_train_length_scan/strip_per_n_no_pooled.png
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS_NO_POOLED = [
    "Global Poisson",
    "Rolling Poisson",
    "Rolling Seasonal",
    "Personalized Gamma-Poisson",
    "Scaled-baseline Hawkes",
    "Joint Hawkes (λ_u + α)",
    "GBDT (experimental)",
]

MODEL_COLORS = {
    "Global Poisson": "#9CA3AF",
    "Rolling Poisson": "#6B7280",
    "Rolling Seasonal": "#374151",
    "Personalized Gamma-Poisson": "#2E5EAA",
    "Scaled-baseline Hawkes": "#0B3C5D",
    "Joint Hawkes (λ_u + α)": "#7B3FAA",
    "GBDT (experimental)": "#D2691E",
}

LABEL_SHORT = {
    "Global Poisson": "Global",
    "Rolling Poisson": "Rolling",
    "Rolling Seasonal": "RS",
    "Personalized Gamma-Poisson": "Pers GP",
    "Scaled-baseline Hawkes": "Scaled H",
    "Joint Hawkes (λ_u + α)": "Joint H",
    "GBDT (experimental)": "GBDT",
}


def plot_cv_strip(merged: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    rng = np.random.default_rng(0)
    # blue for probabilistic models, orange for GBDT (last)
    colors = ["#2E5EAA"] * (len(MODEL_LABELS_NO_POOLED) - 1) + ["#D2691E"]

    for i, label in enumerate(MODEL_LABELS_NO_POOLED):
        vals = merged[label].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        x_jitter = i + (rng.random(len(vals)) - 0.5) * 0.32
        ax.scatter(x_jitter, vals, color=colors[i], alpha=0.65, s=42,
                   edgecolors="white", linewidths=0.6)
        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        ax.hlines(mean_val, i - 0.28, i + 0.28, color="#0B3C5D", linewidth=2.4,
                  label="Mean" if i == 0 else None, zorder=4)
        ax.hlines(median_val, i - 0.28, i + 0.28, color="#D2691E", linewidth=1.8,
                  linestyles="--", label="Median" if i == 0 else None, zorder=4)
        ax.text(i, mean_val, f"{mean_val:.4f}", ha="center", va="bottom",
                fontsize=9, color="#0B3C5D", fontweight="bold")

    ax.set_xticks(range(len(MODEL_LABELS_NO_POOLED)))
    ax.set_xticklabels([lbl.replace(" ", "\n", 1) for lbl in MODEL_LABELS_NO_POOLED], fontsize=8)
    ax.set_ylabel("Test NLL per user-day (lower is better)")
    ax.set_title(f"Blockwise 3-week CV: per-block NLL across {len(merged)} blocks (14d train / 7d test)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_mean_nll_vs_n(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    n_grid = sorted(df["n_days"].unique())
    for label in MODEL_LABELS_NO_POOLED:
        if label not in df.columns:
            continue
        means = []
        for n in n_grid:
            sub = df[df["n_days"] == n][label].dropna().to_numpy(dtype=float)
            means.append(np.mean(sub) if len(sub) else np.nan)
        ax.plot(n_grid, means, marker="o", linewidth=1.8,
                color=MODEL_COLORS[label], label=label)
    ax.set_xlabel("n (длина окна, дней)")
    ax.set_ylabel("Mean test NLL per user-day (lower is better)")
    ax.set_title("Train-length scan: mean test NLL по моделям от длины окна")
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_strip_per_n(df: pd.DataFrame, out: Path) -> None:
    n_grid = sorted(df["n_days"].unique())
    n_panels = len(n_grid)
    n_cols = 5
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.6 * n_rows), squeeze=False)
    rng = np.random.default_rng(0)
    for panel_idx, n_days in enumerate(n_grid):
        ax = axes[panel_idx // n_cols][panel_idx % n_cols]
        sub_df = df[df["n_days"] == n_days]
        m_runs = len(sub_df)
        for i, label in enumerate(MODEL_LABELS_NO_POOLED):
            if label not in df.columns:
                continue
            vals = sub_df[label].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            x_jitter = i + (rng.random(len(vals)) - 0.5) * 0.32
            ax.scatter(x_jitter, vals, s=18, alpha=0.55, color=MODEL_COLORS[label],
                       edgecolors="white", linewidths=0.4)
            mean_val = float(np.mean(vals))
            ax.hlines(mean_val, i - 0.28, i + 0.28, color="#0B3C5D", linewidth=1.8, zorder=4)
        ax.set_xticks(range(len(MODEL_LABELS_NO_POOLED)))
        ax.set_xticklabels([LABEL_SHORT[lbl] for lbl in MODEL_LABELS_NO_POOLED],
                           rotation=45, ha="right", fontsize=7)
        ax.set_title(f"n = {n_days}d  ({m_runs} runs)", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if panel_idx % n_cols == 0:
            ax.set_ylabel("test NLL", fontsize=9)
    for k in range(n_panels, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")
    fig.suptitle("Train-length scan: per-run test NLL по моделям (одна панель на n)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_alpha_norm_vs_n(df: pd.DataFrame, out: Path) -> None:
    n_grid = sorted(df["n_days"].unique())
    hawkes_models = [
        ("Scaled-baseline Hawkes", "Scaled-baseline Hawkes_alpha_norm"),
        ("Joint Hawkes (λ_u + α)", "Joint Hawkes (λ_u + α)_alpha_norm"),
    ]
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for label, col in hawkes_models:
        if col not in df.columns:
            continue
        means, stds = [], []
        for n in n_grid:
            vals = df[df["n_days"] == n][col].dropna().to_numpy(dtype=float)
            means.append(np.mean(vals) if len(vals) else np.nan)
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        color = MODEL_COLORS[label]
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
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_m_u_mean_vs_n(df: pd.DataFrame, out: Path) -> None:
    n_grid = sorted(df["n_days"].unique())
    multiplier_models = [
        ("Scaled-baseline Hawkes", "Scaled-baseline Hawkes_m_u_mean"),
        ("Joint Hawkes (λ_u + α)", "Joint Hawkes (λ_u + α)_m_u_mean"),
    ]
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for label, col in multiplier_models:
        if col not in df.columns:
            continue
        means, stds = [], []
        for n in n_grid:
            vals = df[df["n_days"] == n][col].dropna().to_numpy(dtype=float)
            means.append(np.mean(vals) if len(vals) else np.nan)
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        color = MODEL_COLORS[label]
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
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    cv_dir = Path("diploma/reports/blockwise_cv")
    scan_dir = Path("diploma/reports/11_train_length_scan")

    cv_merged = pd.read_csv(cv_dir / "cv_results_with_joint.csv")
    plot_cv_strip(cv_merged, cv_dir / "cv_strip_plot_no_pooled.png")
    print(f"Saved {cv_dir / 'cv_strip_plot_no_pooled.png'}")

    scan_df = pd.read_csv(scan_dir / "scan_results.csv")
    plot_mean_nll_vs_n(scan_df, scan_dir / "mean_nll_vs_n_no_pooled.png")
    print(f"Saved {scan_dir / 'mean_nll_vs_n_no_pooled.png'}")
    plot_strip_per_n(scan_df, scan_dir / "strip_per_n_no_pooled.png")
    print(f"Saved {scan_dir / 'strip_per_n_no_pooled.png'}")
    plot_alpha_norm_vs_n(scan_df, scan_dir / "alpha_norm_vs_n_no_pooled.png")
    print(f"Saved {scan_dir / 'alpha_norm_vs_n_no_pooled.png'}")
    plot_m_u_mean_vs_n(scan_df, scan_dir / "m_u_mean_vs_n_no_pooled.png")
    print(f"Saved {scan_dir / 'm_u_mean_vs_n_no_pooled.png'}")


if __name__ == "__main__":
    main()

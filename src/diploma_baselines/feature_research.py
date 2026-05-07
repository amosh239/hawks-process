from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment_utils import _resolve_analysis_window

FEATURE_RESEARCH_NAMES = [
    "searches",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "to_cart",
    "to_ord",
]


TARGET_COL = "to_ord"


def _plot_feature_distribution_grid(
    df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    n_cols = 3
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 10.0))
    axes = axes.reshape(n_rows, n_cols)

    for idx, feature in enumerate(FEATURE_RESEARCH_NAMES):
        ax = axes[idx // n_cols, idx % n_cols]
        values = df[feature].to_numpy(dtype=float)
        cap = int(np.quantile(values, 0.99))
        cap = max(cap, 3)
        clipped = np.clip(values, 0, cap)
        bins = np.arange(-0.5, cap + 1.5, 1.0)

        ax.hist(clipped, bins=bins, color="#2E5EAA", edgecolor="white")
        ax.set_yscale("log")
        feature_label = f"{feature} (target)" if feature == TARGET_COL else feature
        ax.set_title(f"{feature_label}\nnonzero={float((values > 0).mean()):.1%}")
        ax.set_xlabel("Daily count")
        ax.set_ylabel("User-days")

        xticks = list(range(0, min(cap, 8) + 1))
        if cap > 8:
            xticks.append(cap)
        xticks = sorted(set(xticks))
        ax.set_xticks(xticks)
        labels = [str(x) for x in xticks]
        if labels:
            labels[-1] = f"{cap}+"
        ax.set_xticklabels(labels)

    for idx in range(len(FEATURE_RESEARCH_NAMES), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle("Distributions of Hawkes count features on analysis window")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    values = corr_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    im = ax.imshow(values, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(list(corr_df.columns), rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(list(corr_df.index))
    ax.set_title("Pearson correlation between Hawkes count features")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pair_scatter_grid(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    out_path: str | Path,
    sample_size: int = 120_000,
) -> None:
    out_path = Path(out_path)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    axes = axes.reshape(2, 2)

    for idx, (x_col, y_col) in enumerate(pairs):
        ax = axes[idx // 2, idx % 2]
        pair_df = df.loc[(df[x_col] > 0) | (df[y_col] > 0), [x_col, y_col]].copy()
        if len(pair_df) > sample_size:
            pair_df = pair_df.sample(sample_size, random_state=42)

        x = np.log1p(pair_df[x_col].to_numpy(dtype=float))
        y = np.log1p(pair_df[y_col].to_numpy(dtype=float))
        corr = float(df[[x_col, y_col]].corr().iloc[0, 1])
        active_share = float(((df[x_col] > 0) | (df[y_col] > 0)).mean())

        ax.scatter(x, y, s=4, alpha=0.10, color="#2E5EAA", edgecolors="none")
        lim = max(float(np.max(x)), float(np.max(y)))
        ax.plot([0.0, lim], [0.0, lim], linestyle="--", linewidth=0.9, color="#444444")
        ax.set_xlabel(f"log1p({x_col})", fontsize=9)
        ax.set_ylabel(f"log1p({y_col})", fontsize=9)
        ax.set_title(f"{x_col} vs {y_col}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.text(
            0.03,
            0.97,
            f"r={corr:.3f}\nactive={active_share:.1%}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )

    fig.suptitle("Pairwise scatter for selected Hawkes channels", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_hawkes_feature_research(
    data_path: str | Path,
    output_dir: str | Path,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        data_path,
        usecols=["event_date", *FEATURE_RESEARCH_NAMES],
        parse_dates=["event_date"],
    )
    start_ts, end_ts = _resolve_analysis_window(df, analysis_start, analysis_end)
    df = df.loc[(df["event_date"] >= start_ts) & (df["event_date"] <= end_ts), ["event_date", *FEATURE_RESEARCH_NAMES]].copy()

    quantiles = df[list(FEATURE_RESEARCH_NAMES)].quantile([0.9, 0.95, 0.99]).T
    summary_df = pd.DataFrame(
        {
            "feature": FEATURE_RESEARCH_NAMES,
            "role": ["target" if c == TARGET_COL else "feature" for c in FEATURE_RESEARCH_NAMES],
            "mean": [float(df[c].mean()) for c in FEATURE_RESEARCH_NAMES],
            "std": [float(df[c].std()) for c in FEATURE_RESEARCH_NAMES],
            "nonzero_share": [float((df[c] > 0).mean()) for c in FEATURE_RESEARCH_NAMES],
            "p90": [float(quantiles.loc[c, 0.9]) for c in FEATURE_RESEARCH_NAMES],
            "p95": [float(quantiles.loc[c, 0.95]) for c in FEATURE_RESEARCH_NAMES],
            "p99": [float(quantiles.loc[c, 0.99]) for c in FEATURE_RESEARCH_NAMES],
            "max": [float(df[c].max()) for c in FEATURE_RESEARCH_NAMES],
        }
    )
    summary_df.to_csv(output_dir / "feature_summary.csv", index=False)

    corr_df = df[list(FEATURE_RESEARCH_NAMES)].corr().astype(float)
    corr_df.to_csv(output_dir / "feature_correlation.csv")

    _plot_feature_distribution_grid(df, output_dir / "feature_distribution_grid.png")
    _plot_correlation_heatmap(corr_df, output_dir / "feature_correlation_heatmap.png")

    pair_plots = [
        ("search_to_cart", "to_cart"),
        ("search_to_ord", "to_ord"),
        ("searches", "to_cart"),
        ("cat_to_cart", "to_cart"),
    ]
    _plot_pair_scatter_grid(df=df, pairs=pair_plots, out_path=output_dir / "pair_scatter_grid.png")

    top_corr = []
    for i, left in enumerate(FEATURE_RESEARCH_NAMES):
        for right in FEATURE_RESEARCH_NAMES[i + 1:]:
            top_corr.append({"left": left, "right": right, "pearson_corr": float(corr_df.loc[left, right])})
    top_corr = sorted(top_corr, key=lambda row: abs(row["pearson_corr"]), reverse=True)
    top_corr_df = pd.DataFrame(top_corr)
    top_corr_df.to_csv(output_dir / "top_correlations.csv", index=False)

    summary = {
        "data_path": str(Path(data_path)),
        "analysis_window": {"start": str(start_ts.date()), "end": str(end_ts.date())},
        "rows": int(len(df)),
        "feature_names": list(FEATURE_RESEARCH_NAMES),
        "target_col": TARGET_COL,
        "top_correlations": top_corr[:8],
        "feature_summary_csv": str(output_dir / "feature_summary.csv"),
        "feature_correlation_csv": str(output_dir / "feature_correlation.csv"),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_daily_aggregate_hawkes_vs_baseline(
    pred_df: pd.DataFrame,
    split_date: pd.Timestamp,
    target_col: str,
    baseline_col: str,
    model_col: str,
    out_path: str | Path,
    baseline_label: str = "Personalized Poisson",
    model_label: str = "Experimental Hawkes",
    title: str = "Daily aggregate intensity: personalized Poisson vs Hawkes",
) -> None:
    out_path = Path(out_path)
    daily = pred_df.groupby("event_date")[[target_col, baseline_col, model_col]].mean().sort_index()
    train_daily = daily[daily.index <= split_date]
    test_daily = daily[daily.index > split_date]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(train_daily.index, train_daily[target_col], color="#2E8B57", linewidth=1.5, label="Train mean orders/day")
    ax.plot(test_daily.index, test_daily[target_col], color="#D2691E", linewidth=1.5, label="Test mean orders/day")
    ax.plot(daily.index, daily[baseline_col], color="#4C4C4C", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.plot(daily.index, daily[model_col], color="#0B3C5D", linewidth=1.3, label=model_label)
    ax.axvline(pd.Timestamp(split_date) + pd.Timedelta(days=0.5), color="#888888", linestyle=":", linewidth=1.0, label="Test split")
    ax.set_ylabel("Mean purchases per user-day")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hawkes_alpha_heatmap(
    alpha_matrix: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    half_lives: list[float] | tuple[float, ...] | np.ndarray,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    alpha_matrix = np.asarray(alpha_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    im = ax.imshow(alpha_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(half_lives)))
    ax.set_xticklabels([str(x) for x in half_lives])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Half-life, days")
    ax.set_ylabel("Feature")
    ax.set_title("Pooled Hawkes alpha by feature and half-life")
    for i in range(alpha_matrix.shape[0]):
        for j in range(alpha_matrix.shape[1]):
            ax.text(j, i, f"{alpha_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(im, ax=ax, label="alpha")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_user_ll_gain_histogram(
    user_ll_df: pd.DataFrame,
    baseline_col: str,
    model_col: str,
    out_path: str | Path,
    xlabel: str = "Delta user-level test LL (model - personalized Poisson)",
    title: str = "User-level LL gain of experimental model",
) -> None:
    out_path = Path(out_path)
    delta = (user_ll_df[model_col] - user_ll_df[baseline_col]).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(delta, bins=40, color="#0B3C5D", alpha=0.85, edgecolor="white")
    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Users")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_user_scale_histogram(
    scales: np.ndarray,
    out_path: str | Path,
    title: str = "Distribution of fitted user-specific baseline scales",
    xlabel: str = "Fitted user-specific scale",
) -> None:
    out_path = Path(out_path)
    scales = np.asarray(scales, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(scales, bins=40, color="#0B3C5D", alpha=0.85, edgecolor="white")
    ax.axvline(float(np.mean(scales)), color="#D2691E", linestyle="--", linewidth=1.0, label="Mean scale")
    ax.axvline(float(np.median(scales)), color="#2E8B57", linestyle=":", linewidth=1.0, label="Median scale")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Users")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fraction_similarity_and_quality(
    summary_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Hawkes kernel stability on train prefixes",
) -> None:
    out_path = Path(out_path)
    df = summary_df.sort_values("train_fraction").copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.plot(df["train_fraction"], df["relative_l2_to_full"], marker="o", color="#0B3C5D", label="Relative L2 to full")
    ax.plot(df["train_fraction"], df["cosine_to_full"], marker="s", color="#2E8B57", label="Cosine to full")
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("Similarity / distance")
    ax.set_title("Coefficient similarity to full-train fit")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(df["train_fraction"], df["test_delta_poisson_loglik_vs_personalized"], marker="o", color="#D2691E", label="Test delta loglik")
    ax.plot(df["train_fraction"], df["base_scale"], marker="s", color="#4C4C4C", label="Learned scale c")
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("Metric value")
    ax.set_title("Test quality and learned baseline scale")
    ax.legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_kernel_paths_by_fraction(
    coef_df: pd.DataFrame,
    top_kernel_labels: list[str],
    out_path: str | Path,
    title: str = "Top Hawkes kernels by train fraction",
) -> None:
    out_path = Path(out_path)
    df = coef_df.sort_values("train_fraction").copy()

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for label in top_kernel_labels:
        ax.plot(df["train_fraction"], df[label], marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("Coefficient value")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weekly_similarity(
    summary_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Weekly Hawkes kernel stability",
) -> None:
    out_path = Path(out_path)
    df = summary_df.sort_values("window_start").copy()
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.plot(x, df["relative_l2_to_full"], marker="o", color="#0B3C5D", label="Relative L2 to full")
    ax.plot(x, df["cosine_to_full"], marker="s", color="#2E8B57", label="Cosine to full")
    ax.set_xlabel("Train week index")
    ax.set_ylabel("Similarity / distance")
    ax.set_title("Coefficient similarity to full-train fit")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(x, df["test_delta_poisson_loglik_vs_personalized"], marker="o", color="#D2691E", label="Test delta loglik")
    ax.plot(x, df["base_scale"], marker="s", color="#4C4C4C", label="Learned scale c")
    ax.set_xlabel("Train week index")
    ax.set_ylabel("Metric value")
    ax.set_title("Test quality and learned baseline scale")
    ax.legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weekly_top_kernel_heatmap(
    heatmap_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Top Hawkes kernels across train weeks",
) -> None:
    out_path = Path(out_path)
    values = heatmap_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(heatmap_df.shape[1]))
    ax.set_xticklabels(list(heatmap_df.columns), rotation=45, ha="right")
    ax.set_yticks(range(heatmap_df.shape[0]))
    ax.set_yticklabels(list(heatmap_df.index))
    ax.set_xlabel("Train week")
    ax.set_ylabel("Kernel")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Coefficient value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

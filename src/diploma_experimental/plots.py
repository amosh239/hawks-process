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

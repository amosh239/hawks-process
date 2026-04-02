from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_orders_histogram(df: pd.DataFrame, target_col: str, out_path: str | Path) -> None:
    out_path = Path(out_path)
    values = df[target_col].to_numpy(dtype=float)
    clipped = np.clip(values, 0, 6)
    bins = np.arange(-0.5, 7.5, 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(clipped, bins=bins, color="#2E5EAA", edgecolor="white")
    ax.set_xticks(range(0, 7))
    ax.set_xticklabels(["0", "1", "2", "3", "4", "5", "6+"])
    ax.set_xlabel("Daily purchase count")
    ax.set_ylabel("User-days")
    ax.set_yscale("log")
    ax.set_title("Distribution of daily purchases")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_daily_aggregate_full_ts(
    full_df: pd.DataFrame,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    split_date: pd.Timestamp,
    target_col: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    daily = full_df.groupby("event_date")[target_col].mean().sort_index()

    excluded_before_mask = daily.index < analysis_start
    excluded_after_mask = daily.index > analysis_end
    train_mask = (daily.index >= analysis_start) & (daily.index <= split_date)
    test_mask = (daily.index > split_date) & (daily.index <= analysis_end)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if excluded_before_mask.any():
        ax.plot(
            daily.index[excluded_before_mask],
            daily.values[excluded_before_mask],
            label="Excluded period",
            color="#B22222",
            linewidth=1.5,
        )
        ax.axvspan(daily.index.min(), analysis_start, color="#B22222", alpha=0.08)
    if excluded_after_mask.any():
        ax.plot(
            daily.index[excluded_after_mask],
            daily.values[excluded_after_mask],
            color="#B22222",
            linewidth=1.5,
        )
        ax.axvspan(analysis_end, daily.index.max(), color="#B22222", alpha=0.08)
    ax.plot(daily.index[train_mask], daily.values[train_mask], label="Analysis train", color="#2E8B57", linewidth=1.5)
    ax.plot(daily.index[test_mask], daily.values[test_mask], label="Analysis test", color="#D2691E", linewidth=1.5)
    ax.axvline(split_date, color="#888888", linestyle=":", linewidth=1.0, label="Test split")
    ax.set_ylabel("Mean purchases per user-day")
    ax.set_title("Daily aggregate purchase intensity on full calendar")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_daily_aggregate_analysis_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    mu: float,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    train_daily = train_df.groupby("event_date")[target_col].mean().sort_index()
    test_daily = test_df.groupby("event_date")[target_col].mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(train_daily.index, train_daily.values, label="Train mean orders/day", color="#2E8B57", linewidth=1.5)
    ax.plot(test_daily.index, test_daily.values, label="Test mean orders/day", color="#D2691E", linewidth=1.5)
    ax.axhline(mu, color="#222222", linestyle="--", linewidth=1.2, label=f"Poisson mu = {mu:.4f}")
    ax.axvline(test_daily.index.min(), color="#888888", linestyle=":", linewidth=1.0, label="Test split")
    ax.set_ylabel("Mean purchases per user-day")
    ax.set_title("Daily aggregate purchase intensity on analysis window")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_daily_aggregate_analysis_window_with_prediction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    train_daily = train_df.groupby("event_date")[[target_col, pred_col]].mean().sort_index()
    test_daily = test_df.groupby("event_date")[[target_col, pred_col]].mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        train_daily.index,
        train_daily[target_col].values,
        label="Train mean orders/day",
        color="#2E8B57",
        linewidth=1.5,
    )
    ax.plot(
        test_daily.index,
        test_daily[target_col].values,
        label="Test mean orders/day",
        color="#D2691E",
        linewidth=1.5,
    )
    ax.plot(
        train_daily.index,
        train_daily[pred_col].values,
        label="Seasonal prediction",
        color="#1F1F1F",
        linewidth=1.2,
        linestyle="--",
    )
    ax.plot(
        test_daily.index,
        test_daily[pred_col].values,
        color="#1F1F1F",
        linewidth=1.2,
        linestyle="--",
    )
    ax.axvline(test_daily.index.min(), color="#888888", linestyle=":", linewidth=1.0, label="Test split")
    ax.set_ylabel("Mean purchases per user-day")
    ax.set_title("Daily aggregate intensity with seasonal prediction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weekday_profile(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    mu: float,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    train_profile = train_df.groupby("dow")[target_col].mean().reindex(range(7), fill_value=0.0)
    test_profile = test_df.groupby("dow")[target_col].mean().reindex(range(7), fill_value=0.0)

    x = np.arange(7)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, train_profile.values, width=width, label="Train", color="#2E8B57")
    ax.bar(x + width / 2, test_profile.values, width=width, label="Test", color="#D2691E")
    ax.axhline(mu, color="#222222", linestyle="--", linewidth=1.2, label="Global Poisson")
    ax.set_xticks(x)
    ax.set_xticklabels(weekday_names)
    ax.set_ylabel("Mean purchases per user-day")
    ax.set_title("Purchase intensity by weekday")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weekday_profile_with_prediction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    predicted_by_dow: np.ndarray,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    train_profile = train_df.groupby("dow")[target_col].mean().reindex(range(7), fill_value=0.0)
    test_profile = test_df.groupby("dow")[target_col].mean().reindex(range(7), fill_value=0.0)

    x = np.arange(7)
    width = 0.28
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, train_profile.values, width=width, label="Train", color="#2E8B57")
    ax.bar(x, test_profile.values, width=width, label="Test", color="#D2691E")
    ax.bar(x + width, predicted_by_dow, width=width, label="Seasonal prediction", color="#4C4C4C")
    ax.set_xticks(x)
    ax.set_xticklabels(weekday_names)
    ax.set_ylabel("Mean purchases per user-day")
    ax.set_title("Purchase intensity by weekday with seasonal model")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_seasonal_factors(seasonal_profile: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = np.arange(7)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x, seasonal_profile, color="#5B8E7D")
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(weekday_names)
    ax.set_ylabel("Seasonal multiplier")
    ax.set_title("Estimated day-of-week seasonal factors")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weekday_profile_with_series_prediction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    train_profile = train_df.groupby("dow")[target_col].mean().reindex(range(7), fill_value=0.0)
    test_profile = test_df.groupby("dow")[target_col].mean().reindex(range(7), fill_value=0.0)
    train_pred_profile = train_df.groupby("dow")[pred_col].mean().reindex(range(7), fill_value=0.0)
    test_pred_profile = test_df.groupby("dow")[pred_col].mean().reindex(range(7), fill_value=0.0)

    x = np.arange(7)
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 1.5 * width, train_profile.values, width=width, label="Train actual", color="#2E8B57")
    ax.bar(x - 0.5 * width, train_pred_profile.values, width=width, label="Train prediction", color="#1F1F1F")
    ax.bar(x + 0.5 * width, test_profile.values, width=width, label="Test actual", color="#D2691E")
    ax.bar(x + 1.5 * width, test_pred_profile.values, width=width, label="Test prediction", color="#7A7A7A")
    ax.set_xticks(x)
    ax.set_xticklabels(weekday_names)
    ax.set_ylabel("Mean purchases per user-day")
    ax.set_title("Purchase intensity by weekday with model prediction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mu_shrinkage(
    user_stats: pd.DataFrame,
    out_path: str | Path,
    title: str = "Gamma-Poisson shrinkage of user multipliers",
) -> None:
    out_path = Path(out_path)
    x = np.clip(user_stats["mu_mle"].to_numpy(dtype=float), 1e-4, None)
    y = np.clip(user_stats["mu_posterior_mean"].to_numpy(dtype=float), 1e-4, None)
    c = np.log1p(np.clip(user_stats["exposure"].to_numpy(dtype=float), 0.0, None))

    lim_min = min(float(x.min()), float(y.min()))
    lim_max = max(float(x.max()), float(y.max()))

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=12, alpha=0.35, edgecolors="none")
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#444444", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("User multiplier MLE")
    ax.set_ylabel("Posterior mean multiplier")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log(1 + train exposure)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_user_ll_scatter(
    user_ll_df: pd.DataFrame,
    prev_col: str,
    new_col: str,
    out_path: str | Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    out_path = Path(out_path)
    x = user_ll_df[prev_col].to_numpy(dtype=float)
    y = user_ll_df[new_col].to_numpy(dtype=float)

    x_lo, x_hi = np.quantile(x, [0.01, 0.99])
    y_lo, y_hi = np.quantile(y, [0.01, 0.99])
    lim_lo = float(min(x_lo, y_lo))
    lim_hi = float(max(x_hi, y_hi))
    improved_share = float(np.mean(y > x))

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(x, y, s=10, alpha=0.22, color="#2E5EAA", edgecolors="none")
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--", color="#444444", linewidth=1.0)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.text(
        0.03,
        0.97,
        f"share(new > prev) = {improved_share:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_ll_vs_test_purchases(
    user_ll_df: pd.DataFrame,
    prev_col: str,
    new_col: str,
    purchases_col: str,
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    purchases = user_ll_df[purchases_col].to_numpy(dtype=float)
    x = np.clip(purchases + 1.0, 1.0, None)
    delta = (user_ll_df[new_col] - user_ll_df[prev_col]).to_numpy(dtype=float)
    improved = delta > 0.0

    abs_delta = np.abs(delta)
    positive_abs = abs_delta[abs_delta > 0]
    linthresh = float(np.quantile(positive_abs, 0.25)) if positive_abs.size else 0.1
    linthresh = max(linthresh, 0.01)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(
        x[~improved],
        delta[~improved],
        s=11,
        alpha=0.22,
        color="#B04A5A",
        edgecolors="none",
        label="Delta LL <= 0",
    )
    ax.scatter(
        x[improved],
        delta[improved],
        s=11,
        alpha=0.22,
        color="#2E8B57",
        edgecolors="none",
        label="Delta LL > 0",
    )
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlabel("Test purchases + 1")
    ax.set_ylabel("Delta user-level test LL (new - previous)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_user_train_test_scatter(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    train_totals = train_df.groupby("user_id")[target_col].sum()
    test_totals = test_df.groupby("user_id")[target_col].sum()
    paired = pd.concat([train_totals.rename("train"), test_totals.rename("test")], axis=1).fillna(0.0)

    x = paired["test"].to_numpy(dtype=float) + 1.0
    y = paired["train"].to_numpy(dtype=float) + 1.0
    lim_min = 1.0
    lim_max = max(float(x.max()), float(y.max()))

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(x, y, s=10, alpha=0.18, color="#2E5EAA", edgecolors="none")
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#444444", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Test purchases + 1")
    ax.set_ylabel("Train purchases + 1")
    ax.set_title("User-level purchases: train vs test")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_activity_lifetime_histogram(lifetimes_days: pd.Series, out_path: str | Path) -> None:
    out_path = Path(out_path)
    values = lifetimes_days.to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("No lifetimes to plot")

    max_lifetime = int(np.ceil(values.max()))
    bins = np.arange(-0.5, max_lifetime + 10.5, 10.0)
    if bins.size < 3:
        bins = np.arange(-0.5, max_lifetime + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=bins, color="#8B5A2B", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("User active lifetime span, days")
    ax.set_ylabel("Users")
    ax.set_title("Distribution of user active lifetimes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rolling_window_sweep(
    sweep_df: pd.DataFrame,
    out_path: str | Path,
    metric_col: str = "test_poisson_loglik",
) -> None:
    out_path = Path(out_path)
    frame = sweep_df.sort_values("window_size").copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        frame["window_size"].to_numpy(dtype=int),
        frame[metric_col].to_numpy(dtype=float),
        color="#2E5EAA",
        linewidth=1.6,
        marker="o",
        markersize=4.5,
    )

    best_idx = frame[metric_col].idxmax()
    best_row = frame.loc[best_idx]
    ax.scatter(
        [best_row["window_size"]],
        [best_row[metric_col]],
        color="#B22222",
        s=45,
        zorder=3,
        label=f"Best w = {int(best_row['window_size'])}",
    )

    ax.set_xlabel("Trailing window size, days")
    ax.set_ylabel("Test Poisson log-likelihood")
    ax.set_title("Rolling-window sweep on test")
    ax.set_xticks(frame["window_size"].to_numpy(dtype=int))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_first_purchase_intensity(
    first_purchase_daily: pd.Series,
    out_path: str | Path,
    title: str = "New users by first purchase date",
    bar_label: str = "Daily first purchases",
) -> None:
    out_path = Path(out_path)
    series = first_purchase_daily.sort_index()
    rolling = series.rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(series.index, series.values, width=1.0, color="#D0D7E2", label=bar_label)
    ax.plot(series.index, rolling.values, color="#C44E52", linewidth=1.8, label="7-day moving average")
    ax.set_ylabel("Users")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DAYUSES_ACTIVITY_COLS, filter_date_range, load_daily_grid, split_panel_by_date
from .metrics import aggregate_user_loglik, evaluate_count_forecast
from .models import (
    GlobalPoissonModel,
    PersonalizedGammaPoissonScaler,
    GlobalRollingMeanPoissonModel,
    GlobalRollingSeasonalPoissonModel,
    GlobalSeasonalPoissonModel,
)
from .plots import (
    plot_activity_lifetime_histogram,
    plot_daily_aggregate_analysis_window,
    plot_daily_aggregate_analysis_window_with_prediction,
    plot_daily_aggregate_full_ts,
    plot_delta_ll_vs_test_purchases,
    plot_first_purchase_intensity,
    plot_orders_histogram,
    plot_mu_shrinkage,
    plot_rolling_window_sweep,
    plot_seasonal_factors,
    plot_user_ll_scatter,
    plot_user_train_test_scatter,
    plot_weekday_profile,
    plot_weekday_profile_with_prediction,
    plot_weekday_profile_with_series_prediction,
)


def _panel_stats(df: pd.DataFrame, target_col: str) -> dict[str, float | int | str]:
    y = df[target_col].astype(float)
    return {
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "date_min": str(df["event_date"].min().date()),
        "date_max": str(df["event_date"].max().date()),
        "mean_target": float(y.mean()),
        "share_nonzero_days": float((y > 0).mean()),
    }


def _resolve_analysis_window(
    df: pd.DataFrame,
    analysis_start: str | None,
    analysis_end: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    available_start = pd.Timestamp(df["event_date"].min())
    available_end = pd.Timestamp(df["event_date"].max())
    start = max(pd.Timestamp(analysis_start), available_start) if analysis_start else available_start
    end = min(pd.Timestamp(analysis_end), available_end) if analysis_end else available_end
    if start > end:
        raise ValueError("Resolved analysis window is empty")
    return start, end


def _activity_lifetimes(activity_df: pd.DataFrame) -> pd.Series:
    activity_cols = [col for col in activity_df.columns if col not in {"user_id", "event_date", "dow"}]
    active_mask = activity_df[activity_cols].sum(axis=1) > 0
    active_days = activity_df.loc[active_mask, ["user_id", "event_date"]]
    spans = active_days.groupby("user_id")["event_date"].agg(["min", "max"])
    return (spans["max"] - spans["min"]).dt.days.rename("lifetime_days")


def _first_purchase_series(df: pd.DataFrame, target_col: str) -> pd.Series:
    positive = df[df[target_col] > 0]
    first_purchase = positive.groupby("user_id")["event_date"].min()
    return first_purchase.value_counts().sort_index()


def _first_seen_series(df: pd.DataFrame) -> pd.Series:
    first_seen = df.groupby("user_id")["event_date"].min()
    return first_seen.value_counts().sort_index()


def _user_purchase_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> dict[str, float | int]:
    train_totals = train_df.groupby("user_id")[target_col].sum()
    test_totals = test_df.groupby("user_id")[target_col].sum()
    paired = pd.concat([train_totals.rename("train"), test_totals.rename("test")], axis=1).fillna(0.0)
    train_pos = paired["train"] > 0
    test_pos = paired["test"] > 0

    both = train_pos & test_pos
    only_train = train_pos & ~test_pos
    only_test = ~train_pos & test_pos
    neither = ~train_pos & ~test_pos

    return {
        "users_total": int(len(paired)),
        "both_count": int(both.sum()),
        "both_share_all": float(both.mean()),
        "only_train_count": int(only_train.sum()),
        "only_train_share_all": float(only_train.mean()),
        "only_test_count": int(only_test.sum()),
        "only_test_share_all": float(only_test.mean()),
        "neither_count": int(neither.sum()),
        "neither_share_all": float(neither.mean()),
    }


def run_global_poisson_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "to_ord",
    train_ratio: float = 0.8,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_target_df = load_daily_grid(data_path, value_cols=[target_col])
    analysis_start_ts, analysis_end_ts = _resolve_analysis_window(full_target_df, analysis_start, analysis_end)
    full_daily_mean = full_target_df.groupby("event_date")[target_col].mean().sort_index()

    df = filter_date_range(full_target_df, start_date=analysis_start_ts, end_date=analysis_end_ts)
    split = split_panel_by_date(df, train_ratio=train_ratio)

    model = GlobalPoissonModel().fit(split.train[target_col].to_numpy())
    train_pred = model.predict(len(split.train))
    test_pred = model.predict(len(split.test))

    train_metrics = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred)
    test_metrics = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred)

    plot_orders_histogram(df, target_col=target_col, out_path=output_dir / "orders_histogram_logy.png")
    plot_daily_aggregate_full_ts(
        full_df=full_target_df,
        analysis_start=analysis_start_ts,
        analysis_end=analysis_end_ts,
        split_date=split.test["event_date"].min(),
        target_col=target_col,
        out_path=output_dir / "daily_aggregate_full_ts.png",
    )
    plot_daily_aggregate_analysis_window(
        train_df=split.train,
        test_df=split.test,
        target_col=target_col,
        mu=model.mu_,
        out_path=output_dir / "daily_aggregate_analysis_window.png",
    )
    plot_weekday_profile(
        train_df=split.train,
        test_df=split.test,
        target_col=target_col,
        mu=model.mu_,
        out_path=output_dir / "weekday_profile.png",
    )
    plot_user_train_test_scatter(
        train_df=split.train,
        test_df=split.test,
        target_col=target_col,
        out_path=output_dir / "user_train_test_scatter.png",
    )

    activity_df = load_daily_grid(data_path, value_cols=DAYUSES_ACTIVITY_COLS)
    activity_df = filter_date_range(activity_df, start_date=analysis_start_ts, end_date=analysis_end_ts)
    lifetimes = _activity_lifetimes(activity_df)
    plot_activity_lifetime_histogram(
        lifetimes_days=lifetimes,
        out_path=output_dir / "activity_lifetime_histogram_logy.png",
    )

    first_purchase_daily = _first_purchase_series(full_target_df, target_col=target_col)
    first_seen_daily = _first_seen_series(full_target_df)
    analysis_dates = pd.date_range(analysis_start_ts, analysis_end_ts, freq="D")
    first_purchase_daily = first_purchase_daily.reindex(analysis_dates, fill_value=0.0)
    first_seen_daily = first_seen_daily.reindex(analysis_dates, fill_value=0.0)
    plot_first_purchase_intensity(
        first_purchase_daily=first_purchase_daily,
        out_path=output_dir / "first_purchase_intensity.png",
    )
    plot_first_purchase_intensity(
        first_purchase_daily=first_seen_daily,
        out_path=output_dir / "first_seen_intensity.png",
        title="New users by first appearance date",
        bar_label="Daily first appearances",
    )

    train_daily = split.train.groupby("event_date")[target_col].mean().reset_index(name="mean_target")
    test_daily = split.test.groupby("event_date")[target_col].mean().reset_index(name="mean_target")
    train_daily["split"] = "train"
    test_daily["split"] = "test"
    daily_summary = pd.concat([train_daily, test_daily], ignore_index=True)
    daily_summary["poisson_prediction"] = float(model.mu_)
    daily_summary.to_csv(output_dir / "daily_mean_summary.csv", index=False)
    user_purchase_overlap = _user_purchase_overlap(split.train, split.test, target_col)

    summary = {
        "data_path": str(Path(data_path)),
        "target_col": target_col,
        "train_ratio": float(train_ratio),
        "requested_analysis_window": {
            "start": str(pd.Timestamp(analysis_start).date()) if analysis_start else None,
            "end": str(pd.Timestamp(analysis_end).date()) if analysis_end else None,
        },
        "resolved_analysis_window": {
            "start": str(analysis_start_ts.date()),
            "end": str(analysis_end_ts.date()),
        },
        "split_date": str(split.split_date.date()),
        "model": "global_poisson",
        "params": model.get_params(),
        "full_panel": _panel_stats(full_target_df, target_col),
        "analysis_panel": _panel_stats(df, target_col),
        "train_panel": _panel_stats(split.train, target_col),
        "test_panel": _panel_stats(split.test, target_col),
        "excluded_periods": {
            "pre_window_mean_target": float(
                full_target_df.loc[full_target_df["event_date"] < analysis_start_ts, target_col].mean()
            ),
            "analysis_window_mean_target": float(df[target_col].mean()),
            "post_window_mean_target": float(
                full_target_df.loc[full_target_df["event_date"] > analysis_end_ts, target_col].mean()
            ),
        },
        "activity_lifetime_days": {
            "mean": float(lifetimes.mean()),
            "median": float(lifetimes.median()),
            "max": float(lifetimes.max()),
        },
        "first_purchase_intensity": {
            "mean_daily_new_users": float(first_purchase_daily.mean()),
            "max_daily_new_users": float(first_purchase_daily.max()),
            "peak_date": str(first_purchase_daily.idxmax().date()),
        },
        "first_seen_intensity": {
            "mean_daily_new_users": float(first_seen_daily.mean()),
            "max_daily_new_users": float(first_seen_daily.max()),
            "peak_date": str(first_seen_daily.idxmax().date()),
        },
        "user_purchase_overlap": user_purchase_overlap,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    excluded_before_daily = full_daily_mean.loc[full_daily_mean.index < analysis_start_ts]
    excluded_after_daily = full_daily_mean.loc[full_daily_mean.index > analysis_end_ts]
    analysis_daily = full_daily_mean.loc[
        (full_daily_mean.index >= analysis_start_ts) & (full_daily_mean.index <= analysis_end_ts)
    ]
    if not excluded_before_daily.empty:
        summary["excluded_periods"]["pre_window_daily_mean_target"] = float(excluded_before_daily.mean())
    if not excluded_after_daily.empty:
        summary["excluded_periods"]["post_window_daily_mean_target"] = float(excluded_after_daily.mean())
    if not analysis_daily.empty:
        summary["excluded_periods"]["analysis_window_daily_mean_target"] = float(analysis_daily.mean())
    if summary["excluded_periods"]["analysis_window_mean_target"] > 0:
        summary["excluded_periods"]["pre_to_analysis_ratio"] = (
            summary["excluded_periods"]["pre_window_mean_target"]
            / summary["excluded_periods"]["analysis_window_mean_target"]
        )
        summary["excluded_periods"]["post_to_analysis_ratio"] = (
            summary["excluded_periods"]["post_window_mean_target"]
            / summary["excluded_periods"]["analysis_window_mean_target"]
        )
    if summary["excluded_periods"].get("analysis_window_daily_mean_target", 0.0) > 0:
        if "pre_window_daily_mean_target" in summary["excluded_periods"]:
            summary["excluded_periods"]["pre_to_analysis_daily_ratio"] = (
                summary["excluded_periods"]["pre_window_daily_mean_target"]
                / summary["excluded_periods"]["analysis_window_daily_mean_target"]
            )
        if "post_window_daily_mean_target" in summary["excluded_periods"]:
            summary["excluded_periods"]["post_to_analysis_daily_ratio"] = (
                summary["excluded_periods"]["post_window_daily_mean_target"]
                / summary["excluded_periods"]["analysis_window_daily_mean_target"]
            )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def run_seasonal_poisson_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "to_ord",
    train_ratio: float = 0.8,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_target_df = load_daily_grid(data_path, value_cols=[target_col])
    analysis_start_ts, analysis_end_ts = _resolve_analysis_window(full_target_df, analysis_start, analysis_end)
    df = filter_date_range(full_target_df, start_date=analysis_start_ts, end_date=analysis_end_ts)
    split = split_panel_by_date(df, train_ratio=train_ratio)

    poisson_model = GlobalPoissonModel().fit(split.train[target_col].to_numpy())
    seasonal_model = GlobalSeasonalPoissonModel().fit(
        split.train[target_col].to_numpy(),
        split.train["dow"].to_numpy(),
    )

    train_pred_poisson = poisson_model.predict(len(split.train))
    test_pred_poisson = poisson_model.predict(len(split.test))
    train_pred_seasonal = seasonal_model.predict(split.train["dow"].to_numpy())
    test_pred_seasonal = seasonal_model.predict(split.test["dow"].to_numpy())

    train_metrics_poisson = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_poisson)
    test_metrics_poisson = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_poisson)
    train_metrics_seasonal = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_seasonal)
    test_metrics_seasonal = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_seasonal)

    train_with_pred = split.train.copy()
    test_with_pred = split.test.copy()
    train_with_pred["seasonal_prediction"] = train_pred_seasonal
    test_with_pred["seasonal_prediction"] = test_pred_seasonal

    predicted_by_dow = seasonal_model.predict(np.arange(7, dtype=int))
    plot_daily_aggregate_analysis_window_with_prediction(
        train_df=train_with_pred,
        test_df=test_with_pred,
        target_col=target_col,
        pred_col="seasonal_prediction",
        out_path=output_dir / "daily_aggregate_analysis_window.png",
    )
    plot_weekday_profile_with_prediction(
        train_df=split.train,
        test_df=split.test,
        target_col=target_col,
        predicted_by_dow=predicted_by_dow,
        out_path=output_dir / "weekday_profile_with_prediction.png",
    )
    plot_seasonal_factors(
        seasonal_profile=seasonal_model.seasonal_profile_,
        out_path=output_dir / "seasonal_factors.png",
    )

    daily_train = train_with_pred.groupby("event_date")[[target_col, "seasonal_prediction"]].mean().reset_index()
    daily_test = test_with_pred.groupby("event_date")[[target_col, "seasonal_prediction"]].mean().reset_index()
    daily_train["split"] = "train"
    daily_test["split"] = "test"
    daily_summary = pd.concat([daily_train, daily_test], ignore_index=True)
    daily_summary.to_csv(output_dir / "daily_mean_summary.csv", index=False)

    summary = {
        "data_path": str(Path(data_path)),
        "target_col": target_col,
        "train_ratio": float(train_ratio),
        "requested_analysis_window": {
            "start": str(pd.Timestamp(analysis_start).date()) if analysis_start else None,
            "end": str(pd.Timestamp(analysis_end).date()) if analysis_end else None,
        },
        "resolved_analysis_window": {
            "start": str(analysis_start_ts.date()),
            "end": str(analysis_end_ts.date()),
        },
        "split_date": str(split.split_date.date()),
        "model": "global_seasonal_poisson",
        "params": seasonal_model.get_params(),
        "train_panel": _panel_stats(split.train, target_col),
        "test_panel": _panel_stats(split.test, target_col),
        "train_metrics_global_poisson": train_metrics_poisson,
        "test_metrics_global_poisson": test_metrics_poisson,
        "train_metrics_seasonal_poisson": train_metrics_seasonal,
        "test_metrics_seasonal_poisson": test_metrics_seasonal,
        "test_improvement_vs_global_poisson": {
            "delta_poisson_loglik": float(
                test_metrics_seasonal["poisson_loglik"] - test_metrics_poisson["poisson_loglik"]
            ),
            "delta_mean_poisson_nll": float(
                test_metrics_seasonal["mean_poisson_nll"] - test_metrics_poisson["mean_poisson_nll"]
            ),
            "delta_mean_poisson_deviance": float(
                test_metrics_seasonal["mean_poisson_deviance"] - test_metrics_poisson["mean_poisson_deviance"]
            ),
            "delta_mae": float(test_metrics_seasonal["mae"] - test_metrics_poisson["mae"]),
            "delta_rmse": float(test_metrics_seasonal["rmse"] - test_metrics_poisson["rmse"]),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def run_rolling_poisson_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "to_ord",
    train_ratio: float = 0.8,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
    window_size: int = 7,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_target_df = load_daily_grid(data_path, value_cols=[target_col])
    analysis_start_ts, analysis_end_ts = _resolve_analysis_window(full_target_df, analysis_start, analysis_end)
    df = filter_date_range(full_target_df, start_date=analysis_start_ts, end_date=analysis_end_ts)
    split = split_panel_by_date(df, train_ratio=train_ratio)

    daily_mean_full = full_target_df.groupby("event_date")[target_col].mean().sort_index()
    rolling_model = GlobalRollingMeanPoissonModel(window_size=int(window_size), min_periods=1).fit(daily_mean_full)
    global_poisson = GlobalPoissonModel().fit(split.train[target_col].to_numpy())

    train_pred_poisson = global_poisson.predict(len(split.train))
    test_pred_poisson = global_poisson.predict(len(split.test))

    train_daily_pred = rolling_model.predict_for_dates(split.train["event_date"])
    test_daily_pred = rolling_model.predict_for_dates(split.test["event_date"])
    train_pred_rolling = train_daily_pred.to_numpy(dtype=float)
    test_pred_rolling = test_daily_pred.to_numpy(dtype=float)

    train_metrics_poisson = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_poisson)
    test_metrics_poisson = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_poisson)
    train_metrics_rolling = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_rolling)
    test_metrics_rolling = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_rolling)

    train_with_pred = split.train.copy()
    test_with_pred = split.test.copy()
    train_with_pred["rolling_prediction"] = train_pred_rolling
    test_with_pred["rolling_prediction"] = test_pred_rolling

    plot_daily_aggregate_analysis_window_with_prediction(
        train_df=train_with_pred,
        test_df=test_with_pred,
        target_col=target_col,
        pred_col="rolling_prediction",
        out_path=output_dir / "daily_aggregate_analysis_window.png",
    )

    daily_train = train_with_pred.groupby("event_date")[[target_col, "rolling_prediction"]].mean().reset_index()
    daily_test = test_with_pred.groupby("event_date")[[target_col, "rolling_prediction"]].mean().reset_index()
    daily_train["split"] = "train"
    daily_test["split"] = "test"
    daily_summary = pd.concat([daily_train, daily_test], ignore_index=True)
    daily_summary.to_csv(output_dir / "daily_mean_summary.csv", index=False)

    sweep_rows: list[dict[str, float | int]] = []
    y_train = split.train[target_col].to_numpy()
    y_test = split.test[target_col].to_numpy()
    for sweep_w in range(1, 11):
        sweep_model = GlobalRollingMeanPoissonModel(window_size=int(sweep_w), min_periods=1).fit(daily_mean_full)
        sweep_train_pred = sweep_model.predict_for_dates(split.train["event_date"]).to_numpy(dtype=float)
        sweep_test_pred = sweep_model.predict_for_dates(split.test["event_date"]).to_numpy(dtype=float)
        sweep_train_metrics = evaluate_count_forecast(y_train, sweep_train_pred)
        sweep_test_metrics = evaluate_count_forecast(y_test, sweep_test_pred)
        sweep_rows.append(
            {
                "window_size": int(sweep_w),
                "train_poisson_loglik": float(sweep_train_metrics["poisson_loglik"]),
                "test_poisson_loglik": float(sweep_test_metrics["poisson_loglik"]),
                "train_mean_poisson_nll": float(sweep_train_metrics["mean_poisson_nll"]),
                "test_mean_poisson_nll": float(sweep_test_metrics["mean_poisson_nll"]),
                "train_mean_prediction": float(sweep_train_metrics["mean_prediction"]),
                "test_mean_prediction": float(sweep_test_metrics["mean_prediction"]),
            }
        )
    sweep_df = pd.DataFrame(sweep_rows).sort_values("window_size").reset_index(drop=True)
    sweep_df.to_csv(output_dir / "rolling_window_sweep.csv", index=False)
    plot_rolling_window_sweep(
        sweep_df=sweep_df,
        out_path=output_dir / "rolling_window_sweep_test_loglik.png",
        metric_col="test_poisson_loglik",
    )
    best_row = sweep_df.loc[sweep_df["test_poisson_loglik"].idxmax()]

    summary = {
        "data_path": str(Path(data_path)),
        "target_col": target_col,
        "train_ratio": float(train_ratio),
        "requested_analysis_window": {
            "start": str(pd.Timestamp(analysis_start).date()) if analysis_start else None,
            "end": str(pd.Timestamp(analysis_end).date()) if analysis_end else None,
        },
        "resolved_analysis_window": {
            "start": str(analysis_start_ts.date()),
            "end": str(analysis_end_ts.date()),
        },
        "split_date": str(split.split_date.date()),
        "model": "global_rolling_mean_poisson",
        "params": rolling_model.get_params(),
        "train_panel": _panel_stats(split.train, target_col),
        "test_panel": _panel_stats(split.test, target_col),
        "train_metrics_global_poisson": train_metrics_poisson,
        "test_metrics_global_poisson": test_metrics_poisson,
        "train_metrics_rolling_poisson": train_metrics_rolling,
        "test_metrics_rolling_poisson": test_metrics_rolling,
        "test_improvement_vs_global_poisson": {
            "delta_poisson_loglik": float(
                test_metrics_rolling["poisson_loglik"] - test_metrics_poisson["poisson_loglik"]
            ),
            "delta_mean_poisson_nll": float(
                test_metrics_rolling["mean_poisson_nll"] - test_metrics_poisson["mean_poisson_nll"]
            ),
            "delta_mean_poisson_deviance": float(
                test_metrics_rolling["mean_poisson_deviance"] - test_metrics_poisson["mean_poisson_deviance"]
            ),
            "delta_mae": float(test_metrics_rolling["mae"] - test_metrics_poisson["mae"]),
            "delta_rmse": float(test_metrics_rolling["rmse"] - test_metrics_poisson["rmse"]),
        },
        "rolling_window_sweep": {
            "window_sizes": sweep_df["window_size"].astype(int).tolist(),
            "test_poisson_loglik": sweep_df["test_poisson_loglik"].astype(float).tolist(),
            "best_window_size_by_test_poisson_loglik": int(best_row["window_size"]),
            "best_test_poisson_loglik": float(best_row["test_poisson_loglik"]),
            "selected_window_size": int(window_size),
            "selected_test_poisson_loglik": float(
                sweep_df.loc[sweep_df["window_size"] == int(window_size), "test_poisson_loglik"].iloc[0]
            ),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def run_rolling_seasonal_poisson_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "to_ord",
    train_ratio: float = 0.8,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
    window_size: int = 7,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_target_df = load_daily_grid(data_path, value_cols=[target_col])
    analysis_start_ts, analysis_end_ts = _resolve_analysis_window(full_target_df, analysis_start, analysis_end)
    df = filter_date_range(full_target_df, start_date=analysis_start_ts, end_date=analysis_end_ts)
    split = split_panel_by_date(df, train_ratio=train_ratio)
    daily_mean_full = full_target_df.groupby("event_date")[target_col].mean().sort_index()
    train_daily_mean = split.train.groupby("event_date")[target_col].mean().sort_index()

    global_poisson = GlobalPoissonModel().fit(split.train[target_col].to_numpy())
    rolling_poisson = GlobalRollingMeanPoissonModel(window_size=int(window_size), min_periods=1).fit(daily_mean_full)
    rolling_seasonal = GlobalRollingSeasonalPoissonModel(window_size=int(window_size), min_periods=1).fit(
        train_daily_mean,
        daily_mean_full,
    )

    train_pred_poisson = global_poisson.predict(len(split.train))
    test_pred_poisson = global_poisson.predict(len(split.test))

    train_pred_rolling = rolling_poisson.predict_for_dates(split.train["event_date"]).to_numpy(dtype=float)
    test_pred_rolling = rolling_poisson.predict_for_dates(split.test["event_date"]).to_numpy(dtype=float)

    train_pred_combo = rolling_seasonal.predict_for_dates(split.train["event_date"]).to_numpy(dtype=float)
    test_pred_combo = rolling_seasonal.predict_for_dates(split.test["event_date"]).to_numpy(dtype=float)

    train_metrics_poisson = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_poisson)
    test_metrics_poisson = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_poisson)
    train_metrics_rolling = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_rolling)
    test_metrics_rolling = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_rolling)
    train_metrics_combo = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_combo)
    test_metrics_combo = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_combo)

    train_with_pred = split.train.copy()
    test_with_pred = split.test.copy()
    train_with_pred["rolling_seasonal_prediction"] = train_pred_combo
    test_with_pred["rolling_seasonal_prediction"] = test_pred_combo

    plot_daily_aggregate_analysis_window_with_prediction(
        train_df=train_with_pred,
        test_df=test_with_pred,
        target_col=target_col,
        pred_col="rolling_seasonal_prediction",
        out_path=output_dir / "daily_aggregate_analysis_window.png",
    )
    plot_weekday_profile_with_series_prediction(
        train_df=train_with_pred,
        test_df=test_with_pred,
        target_col=target_col,
        pred_col="rolling_seasonal_prediction",
        out_path=output_dir / "weekday_profile_with_prediction.png",
    )
    plot_seasonal_factors(
        seasonal_profile=rolling_seasonal.seasonal_profile_,
        out_path=output_dir / "seasonal_factors.png",
    )

    daily_train = train_with_pred.groupby("event_date")[[target_col, "rolling_seasonal_prediction"]].mean().reset_index()
    daily_test = test_with_pred.groupby("event_date")[[target_col, "rolling_seasonal_prediction"]].mean().reset_index()
    daily_train["split"] = "train"
    daily_test["split"] = "test"
    daily_summary = pd.concat([daily_train, daily_test], ignore_index=True)
    daily_summary.to_csv(output_dir / "daily_mean_summary.csv", index=False)

    summary = {
        "data_path": str(Path(data_path)),
        "target_col": target_col,
        "train_ratio": float(train_ratio),
        "requested_analysis_window": {
            "start": str(pd.Timestamp(analysis_start).date()) if analysis_start else None,
            "end": str(pd.Timestamp(analysis_end).date()) if analysis_end else None,
        },
        "resolved_analysis_window": {
            "start": str(analysis_start_ts.date()),
            "end": str(analysis_end_ts.date()),
        },
        "split_date": str(split.split_date.date()),
        "model": "global_rolling_seasonal_poisson",
        "params": rolling_seasonal.get_params(),
        "train_panel": _panel_stats(split.train, target_col),
        "test_panel": _panel_stats(split.test, target_col),
        "train_metrics_global_poisson": train_metrics_poisson,
        "test_metrics_global_poisson": test_metrics_poisson,
        "train_metrics_rolling_poisson": train_metrics_rolling,
        "test_metrics_rolling_poisson": test_metrics_rolling,
        "train_metrics_rolling_seasonal_poisson": train_metrics_combo,
        "test_metrics_rolling_seasonal_poisson": test_metrics_combo,
        "test_improvement_vs_global_poisson": {
            "delta_poisson_loglik": float(
                test_metrics_combo["poisson_loglik"] - test_metrics_poisson["poisson_loglik"]
            ),
            "delta_mean_poisson_nll": float(
                test_metrics_combo["mean_poisson_nll"] - test_metrics_poisson["mean_poisson_nll"]
            ),
            "delta_mean_poisson_deviance": float(
                test_metrics_combo["mean_poisson_deviance"] - test_metrics_poisson["mean_poisson_deviance"]
            ),
            "delta_mae": float(test_metrics_combo["mae"] - test_metrics_poisson["mae"]),
            "delta_rmse": float(test_metrics_combo["rmse"] - test_metrics_poisson["rmse"]),
        },
        "test_improvement_vs_rolling_poisson": {
            "delta_poisson_loglik": float(
                test_metrics_combo["poisson_loglik"] - test_metrics_rolling["poisson_loglik"]
            ),
            "delta_mean_poisson_nll": float(
                test_metrics_combo["mean_poisson_nll"] - test_metrics_rolling["mean_poisson_nll"]
            ),
            "delta_mean_poisson_deviance": float(
                test_metrics_combo["mean_poisson_deviance"] - test_metrics_rolling["mean_poisson_deviance"]
            ),
            "delta_mae": float(test_metrics_combo["mae"] - test_metrics_rolling["mae"]),
            "delta_rmse": float(test_metrics_combo["rmse"] - test_metrics_rolling["rmse"]),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def run_personalized_rolling_seasonal_poisson_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "to_ord",
    train_ratio: float = 0.8,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
    window_size: int = 7,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_target_df = load_daily_grid(data_path, value_cols=[target_col])
    analysis_start_ts, analysis_end_ts = _resolve_analysis_window(full_target_df, analysis_start, analysis_end)
    df = filter_date_range(full_target_df, start_date=analysis_start_ts, end_date=analysis_end_ts)
    split = split_panel_by_date(df, train_ratio=train_ratio)
    daily_mean_full = full_target_df.groupby("event_date")[target_col].mean().sort_index()
    train_daily_mean = split.train.groupby("event_date")[target_col].mean().sort_index()

    global_poisson = GlobalPoissonModel().fit(split.train[target_col].to_numpy())
    base_model = GlobalRollingSeasonalPoissonModel(window_size=int(window_size), min_periods=1).fit(
        train_daily_mean,
        daily_mean_full,
    )

    train_base = base_model.predict_for_dates(split.train["event_date"]).to_numpy(dtype=float)
    test_base = base_model.predict_for_dates(split.test["event_date"]).to_numpy(dtype=float)

    scaler = PersonalizedGammaPoissonScaler().fit(
        split.train["user_id"].to_numpy(),
        split.train[target_col].to_numpy(),
        train_base,
    )

    train_pred_global = global_poisson.predict(len(split.train))
    test_pred_global = global_poisson.predict(len(split.test))
    train_pred_base = train_base
    test_pred_base = test_base
    train_pred_mle = scaler.predict(split.train["user_id"].to_numpy(), train_base, method="mle")
    test_pred_mle = scaler.predict(split.test["user_id"].to_numpy(), test_base, method="mle")
    train_pred_post = scaler.predict(split.train["user_id"].to_numpy(), train_base, method="posterior_mean")
    test_pred_post = scaler.predict(split.test["user_id"].to_numpy(), test_base, method="posterior_mean")

    train_metrics_global = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_global)
    test_metrics_global = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_global)
    train_metrics_base = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_base)
    test_metrics_base = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_base)
    train_metrics_mle = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_mle)
    test_metrics_mle = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_mle)
    train_metrics_post = evaluate_count_forecast(split.train[target_col].to_numpy(), train_pred_post)
    test_metrics_post = evaluate_count_forecast(split.test[target_col].to_numpy(), test_pred_post)

    train_with_pred = split.train.copy()
    test_with_pred = split.test.copy()
    train_with_pred["personalized_prediction"] = train_pred_post
    test_with_pred["personalized_prediction"] = test_pred_post

    plot_daily_aggregate_analysis_window_with_prediction(
        train_df=train_with_pred,
        test_df=test_with_pred,
        target_col=target_col,
        pred_col="personalized_prediction",
        out_path=output_dir / "daily_aggregate_analysis_window.png",
    )
    plot_mu_shrinkage(
        user_stats=scaler.user_stats_,
        out_path=output_dir / "mu_shrinkage.png",
    )
    positive_purchase_stats = scaler.user_stats_.loc[scaler.user_stats_["y_sum"] > 0].copy()
    plot_mu_shrinkage(
        user_stats=positive_purchase_stats,
        out_path=output_dir / "mu_shrinkage_positive_only.png",
        title="Gamma-Poisson shrinkage without zero-purchase users",
    )

    daily_train = train_with_pred.groupby("event_date")[[target_col, "personalized_prediction"]].mean().reset_index()
    daily_test = test_with_pred.groupby("event_date")[[target_col, "personalized_prediction"]].mean().reset_index()
    daily_train["split"] = "train"
    daily_test["split"] = "test"
    daily_summary = pd.concat([daily_train, daily_test], ignore_index=True)
    daily_summary.to_csv(output_dir / "daily_mean_summary.csv", index=False)

    unseen_test_users = pd.Index(split.test["user_id"].unique()).difference(pd.Index(scaler.user_stats_.index))
    summary = {
        "data_path": str(Path(data_path)),
        "target_col": target_col,
        "train_ratio": float(train_ratio),
        "requested_analysis_window": {
            "start": str(pd.Timestamp(analysis_start).date()) if analysis_start else None,
            "end": str(pd.Timestamp(analysis_end).date()) if analysis_end else None,
        },
        "resolved_analysis_window": {
            "start": str(analysis_start_ts.date()),
            "end": str(analysis_end_ts.date()),
        },
        "split_date": str(split.split_date.date()),
        "model": "personalized_rolling_seasonal_poisson",
        "base_model": base_model.get_params(),
        "gamma_prior": scaler.get_params(),
        "train_panel": _panel_stats(split.train, target_col),
        "test_panel": _panel_stats(split.test, target_col),
        "train_metrics_global_poisson": train_metrics_global,
        "test_metrics_global_poisson": test_metrics_global,
        "train_metrics_base_rolling_seasonal_poisson": train_metrics_base,
        "test_metrics_base_rolling_seasonal_poisson": test_metrics_base,
        "train_metrics_mle_personalized": train_metrics_mle,
        "test_metrics_mle_personalized": test_metrics_mle,
        "train_metrics_posterior_personalized": train_metrics_post,
        "test_metrics_posterior_personalized": test_metrics_post,
        "test_improvement_vs_base": {
            "delta_poisson_loglik": float(test_metrics_post["poisson_loglik"] - test_metrics_base["poisson_loglik"]),
            "delta_mean_poisson_nll": float(
                test_metrics_post["mean_poisson_nll"] - test_metrics_base["mean_poisson_nll"]
            ),
            "delta_mean_poisson_deviance": float(
                test_metrics_post["mean_poisson_deviance"] - test_metrics_base["mean_poisson_deviance"]
            ),
            "delta_mae": float(test_metrics_post["mae"] - test_metrics_base["mae"]),
            "delta_rmse": float(test_metrics_post["rmse"] - test_metrics_base["rmse"]),
        },
        "test_improvement_vs_mle_personalized": {
            "delta_poisson_loglik": float(test_metrics_post["poisson_loglik"] - test_metrics_mle["poisson_loglik"]),
            "delta_mean_poisson_nll": float(
                test_metrics_post["mean_poisson_nll"] - test_metrics_mle["mean_poisson_nll"]
            ),
            "delta_mean_poisson_deviance": float(
                test_metrics_post["mean_poisson_deviance"] - test_metrics_mle["mean_poisson_deviance"]
            ),
            "delta_mae": float(test_metrics_post["mae"] - test_metrics_mle["mae"]),
            "delta_rmse": float(test_metrics_post["rmse"] - test_metrics_mle["rmse"]),
        },
        "user_multiplier_stats": {
            "users_with_train_history": int(len(scaler.user_stats_)),
            "unseen_test_users": int(len(unseen_test_users)),
            "unseen_test_user_share": float(len(unseen_test_users) / max(split.test["user_id"].nunique(), 1)),
            "zero_train_purchase_users": int((scaler.user_stats_["y_sum"] == 0).sum()),
            "zero_train_purchase_user_share": float((scaler.user_stats_["y_sum"] == 0).mean()),
            "mu_mle_q10": float(scaler.user_stats_["mu_mle"].quantile(0.1)),
            "mu_mle_q50": float(scaler.user_stats_["mu_mle"].quantile(0.5)),
            "mu_mle_q90": float(scaler.user_stats_["mu_mle"].quantile(0.9)),
            "mu_post_q10": float(scaler.user_stats_["mu_posterior_mean"].quantile(0.1)),
            "mu_post_q50": float(scaler.user_stats_["mu_posterior_mean"].quantile(0.5)),
            "mu_post_q90": float(scaler.user_stats_["mu_posterior_mean"].quantile(0.9)),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def _pair_ll_summary(user_ll_df: pd.DataFrame, prev_col: str, new_col: str) -> dict[str, float]:
    delta = user_ll_df[new_col] - user_ll_df[prev_col]
    return {
        "users": int(len(user_ll_df)),
        "share_new_better": float((delta > 0).mean()),
        "mean_delta_ll": float(delta.mean()),
        "median_delta_ll": float(delta.median()),
        "q10_delta_ll": float(delta.quantile(0.1)),
        "q90_delta_ll": float(delta.quantile(0.9)),
    }


def _delta_by_test_purchase_bucket(
    user_ll_df: pd.DataFrame,
    prev_col: str,
    new_col: str,
    purchases_col: str = "test_purchases",
) -> list[dict[str, float | int | str]]:
    delta = user_ll_df[new_col] - user_ll_df[prev_col]
    frame = user_ll_df.copy()
    frame["delta_ll"] = delta

    bucket_defs = [
        (0.0, 0.0, "0"),
        (1.0, 1.0, "1"),
        (2.0, 2.0, "2"),
        (3.0, 5.0, "3-5"),
        (6.0, 10.0, "6-10"),
        (11.0, np.inf, "11+"),
    ]
    rows: list[dict[str, float | int | str]] = []
    total_users = max(len(frame), 1)
    for lo, hi, label in bucket_defs:
        if np.isinf(hi):
            bucket = frame[frame[purchases_col] >= lo]
        else:
            bucket = frame[(frame[purchases_col] >= lo) & (frame[purchases_col] <= hi)]
        if bucket.empty:
            continue
        rows.append(
            {
                "bucket": label,
                "users": int(len(bucket)),
                "share_users": float(len(bucket) / total_users),
                "share_new_better": float((bucket["delta_ll"] > 0).mean()),
                "mean_delta_ll": float(bucket["delta_ll"].mean()),
                "median_delta_ll": float(bucket["delta_ll"].median()),
            }
        )
    return rows


def run_user_level_ll_diagnostics(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "to_ord",
    train_ratio: float = 0.8,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
    window_size: int = 7,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_target_df = load_daily_grid(data_path, value_cols=[target_col])
    analysis_start_ts, analysis_end_ts = _resolve_analysis_window(full_target_df, analysis_start, analysis_end)
    df = filter_date_range(full_target_df, start_date=analysis_start_ts, end_date=analysis_end_ts)
    split = split_panel_by_date(df, train_ratio=train_ratio)
    daily_mean_full = full_target_df.groupby("event_date")[target_col].mean().sort_index()
    train_daily_mean = split.train.groupby("event_date")[target_col].mean().sort_index()

    global_poisson = GlobalPoissonModel().fit(split.train[target_col].to_numpy())
    rolling = GlobalRollingMeanPoissonModel(window_size=int(window_size), min_periods=1).fit(daily_mean_full)
    rolling_seasonal = GlobalRollingSeasonalPoissonModel(window_size=int(window_size), min_periods=1).fit(
        train_daily_mean,
        daily_mean_full,
    )

    test_pred_global = global_poisson.predict(len(split.test))
    test_pred_rolling = rolling.predict_for_dates(split.test["event_date"]).to_numpy(dtype=float)
    test_pred_rolling_seasonal = rolling_seasonal.predict_for_dates(split.test["event_date"]).to_numpy(dtype=float)
    train_pred_rolling_seasonal = rolling_seasonal.predict_for_dates(split.train["event_date"]).to_numpy(dtype=float)

    personalized = PersonalizedGammaPoissonScaler().fit(
        split.train["user_id"].to_numpy(),
        split.train[target_col].to_numpy(),
        train_pred_rolling_seasonal,
    )
    test_pred_personalized = personalized.predict(
        split.test["user_id"].to_numpy(),
        test_pred_rolling_seasonal,
        method="posterior_mean",
    )

    ll_global = aggregate_user_loglik(
        split.test["user_id"].to_numpy(),
        split.test[target_col].to_numpy(),
        test_pred_global,
        ll_col="ll_global_poisson",
    )
    ll_rolling = aggregate_user_loglik(
        split.test["user_id"].to_numpy(),
        split.test[target_col].to_numpy(),
        test_pred_rolling,
        ll_col="ll_rolling_poisson",
    )
    ll_rolling_seasonal = aggregate_user_loglik(
        split.test["user_id"].to_numpy(),
        split.test[target_col].to_numpy(),
        test_pred_rolling_seasonal,
        ll_col="ll_rolling_seasonal_poisson",
    )
    ll_personalized = aggregate_user_loglik(
        split.test["user_id"].to_numpy(),
        split.test[target_col].to_numpy(),
        test_pred_personalized,
        ll_col="ll_personalized_rolling_seasonal_poisson",
    )
    test_purchases = (
        split.test.groupby("user_id", as_index=False)[target_col]
        .sum()
        .rename(columns={target_col: "test_purchases"})
    )

    user_ll = ll_global.merge(ll_rolling, on="user_id", how="outer").merge(
        ll_rolling_seasonal,
        on="user_id",
        how="outer",
    )
    user_ll = user_ll.merge(ll_personalized, on="user_id", how="outer")
    user_ll = user_ll.merge(test_purchases, on="user_id", how="left")
    user_ll = user_ll.fillna(0.0).sort_values("user_id").reset_index(drop=True)
    user_ll.to_csv(output_dir / "user_ll_scores.csv", index=False)

    plot_user_ll_scatter(
        user_ll_df=user_ll,
        prev_col="ll_global_poisson",
        new_col="ll_rolling_poisson",
        out_path=output_dir / "user_ll_scatter_poisson_vs_rolling.png",
        title="Per-user test LL: global Poisson vs rolling Poisson",
        x_label="Previous model LL: global Poisson",
        y_label="New model LL: rolling Poisson",
    )
    plot_user_ll_scatter(
        user_ll_df=user_ll,
        prev_col="ll_rolling_poisson",
        new_col="ll_rolling_seasonal_poisson",
        out_path=output_dir / "user_ll_scatter_rolling_vs_rolling_seasonal.png",
        title="Per-user test LL: rolling Poisson vs rolling seasonal Poisson",
        x_label="Previous model LL: rolling Poisson",
        y_label="New model LL: rolling seasonal Poisson",
    )
    plot_delta_ll_vs_test_purchases(
        user_ll_df=user_ll,
        prev_col="ll_global_poisson",
        new_col="ll_rolling_poisson",
        purchases_col="test_purchases",
        out_path=output_dir / "delta_ll_vs_test_purchases_poisson_to_rolling.png",
        title="Per-user delta LL vs test purchases: global Poisson -> rolling Poisson",
    )
    plot_delta_ll_vs_test_purchases(
        user_ll_df=user_ll,
        prev_col="ll_rolling_poisson",
        new_col="ll_rolling_seasonal_poisson",
        purchases_col="test_purchases",
        out_path=output_dir / "delta_ll_vs_test_purchases_rolling_to_rolling_seasonal.png",
        title="Per-user delta LL vs test purchases: rolling Poisson -> rolling seasonal Poisson",
    )
    plot_delta_ll_vs_test_purchases(
        user_ll_df=user_ll,
        prev_col="ll_rolling_seasonal_poisson",
        new_col="ll_personalized_rolling_seasonal_poisson",
        purchases_col="test_purchases",
        out_path=output_dir / "delta_ll_vs_test_purchases_rolling_seasonal_to_personalized.png",
        title="Per-user delta LL vs test purchases: rolling seasonal -> personalized rolling seasonal",
    )

    summary = {
        "data_path": str(Path(data_path)),
        "target_col": target_col,
        "train_ratio": float(train_ratio),
        "resolved_analysis_window": {
            "start": str(analysis_start_ts.date()),
            "end": str(analysis_end_ts.date()),
        },
        "split_date": str(split.split_date.date()),
        "poisson_to_rolling": _pair_ll_summary(
            user_ll,
            prev_col="ll_global_poisson",
            new_col="ll_rolling_poisson",
        ),
        "poisson_to_rolling_by_test_purchases": _delta_by_test_purchase_bucket(
            user_ll,
            prev_col="ll_global_poisson",
            new_col="ll_rolling_poisson",
        ),
        "rolling_to_rolling_seasonal": _pair_ll_summary(
            user_ll,
            prev_col="ll_rolling_poisson",
            new_col="ll_rolling_seasonal_poisson",
        ),
        "rolling_to_rolling_seasonal_by_test_purchases": _delta_by_test_purchase_bucket(
            user_ll,
            prev_col="ll_rolling_poisson",
            new_col="ll_rolling_seasonal_poisson",
        ),
        "rolling_seasonal_to_personalized": _pair_ll_summary(
            user_ll,
            prev_col="ll_rolling_seasonal_poisson",
            new_col="ll_personalized_rolling_seasonal_poisson",
        ),
        "rolling_seasonal_to_personalized_by_test_purchases": _delta_by_test_purchase_bucket(
            user_ll,
            prev_col="ll_rolling_seasonal_poisson",
            new_col="ll_personalized_rolling_seasonal_poisson",
        ),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.diploma_baselines.data import filter_date_range, load_daily_grid, split_panel_by_date
from src.diploma_baselines.experiment_utils import (
    _delta_by_test_purchase_bucket,
    _pair_ll_summary,
    _panel_stats,
    _resolve_analysis_window,
)
from src.diploma_baselines.metrics import aggregate_user_loglik, evaluate_count_forecast
from src.diploma_baselines.models.personalized_gamma_poisson import PersonalizedGammaPoissonScaler
from src.diploma_baselines.models.rolling_seasonal_poisson import GlobalRollingSeasonalPoissonModel
from src.diploma_baselines.plots import (
    plot_daily_aggregate_hawkes_vs_baseline,
    plot_delta_ll_vs_test_purchases,
    plot_user_ll_gain_histogram,
)

from .gbdt import SOURCE_FEATURES, build_feature_tables, fit_global_poisson_gbdt


def run_experimental_2_gbdt(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "to_ord",
    train_ratio: float = 0.8,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
    window_size: int = 7,
    seed: int = 42,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    max_iter: int = 200,
    min_samples_leaf: int = 40,
    source_features: list[str] | tuple[str, ...] | None = None,
    max_users: int | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_features = list(source_features or SOURCE_FEATURES)
    value_cols = list(dict.fromkeys([target_col, *source_features]))
    full_df = load_daily_grid(data_path, value_cols=value_cols)
    if max_users is not None:
        keep_users = pd.Index(sorted(full_df["user_id"].drop_duplicates()))[: int(max_users)]
        full_df = full_df[full_df["user_id"].isin(keep_users)].copy()
    analysis_start_ts, analysis_end_ts = _resolve_analysis_window(full_df, analysis_start, analysis_end)
    analysis_df = filter_date_range(full_df, start_date=analysis_start_ts, end_date=analysis_end_ts)
    split = split_panel_by_date(analysis_df, train_ratio=train_ratio)

    daily_mean_full = full_df.groupby("event_date")[target_col].mean().sort_index()
    train_daily_mean = split.train.groupby("event_date")[target_col].mean().sort_index()
    rolling_seasonal = GlobalRollingSeasonalPoissonModel(window_size=int(window_size), min_periods=1).fit(
        train_daily_mean,
        daily_mean_full,
    )
    global_base_analysis = rolling_seasonal.predict_for_dates(analysis_df["event_date"]).to_numpy(dtype=float)
    global_base_train = rolling_seasonal.predict_for_dates(split.train["event_date"]).to_numpy(dtype=float)
    scaler = PersonalizedGammaPoissonScaler().fit(
        split.train["user_id"].to_numpy(),
        split.train[target_col].to_numpy(),
        global_base_train,
    )
    personalized_df = analysis_df.loc[:, ["user_id", "event_date", target_col]].copy()
    personalized_df["personalized_poisson_prediction"] = scaler.predict(
        personalized_df["user_id"].to_numpy(),
        global_base_analysis,
        method="posterior_mean",
    )

    feature_table = build_feature_tables(
        full_df=full_df.loc[:, ["user_id", "event_date", *list(dict.fromkeys([*source_features, target_col]))]].copy(),
        analysis_start=analysis_start_ts,
        analysis_end=analysis_end_ts,
        split_date=split.split_date,
        target_col=target_col,
        source_features=source_features,
    )
    model = fit_global_poisson_gbdt(
        feature_table,
        seed=seed,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
    )
    train_pred = np.clip(model.predict(feature_table.x_train), 1e-8, None)
    test_pred = np.clip(model.predict(feature_table.x_test), 1e-8, None)

    train_pred_df = feature_table.row_index_train.rename(columns={"target": target_col}).copy()
    test_pred_df = feature_table.row_index_test.rename(columns={"target": target_col}).copy()
    train_pred_df["gbdt_prediction"] = train_pred
    test_pred_df["gbdt_prediction"] = test_pred
    train_pred_df = train_pred_df.merge(personalized_df, on=["user_id", "event_date", target_col], how="left")
    test_pred_df = test_pred_df.merge(personalized_df, on=["user_id", "event_date", target_col], how="left")
    pred_df = pd.concat([train_pred_df, test_pred_df], ignore_index=True).sort_values(["user_id", "event_date"]).reset_index(drop=True)

    train_metrics_base = evaluate_count_forecast(
        train_pred_df[target_col].to_numpy(),
        train_pred_df["personalized_poisson_prediction"].to_numpy(),
    )
    test_metrics_base = evaluate_count_forecast(
        test_pred_df[target_col].to_numpy(),
        test_pred_df["personalized_poisson_prediction"].to_numpy(),
    )
    train_metrics_gbdt = evaluate_count_forecast(
        train_pred_df[target_col].to_numpy(),
        train_pred_df["gbdt_prediction"].to_numpy(),
    )
    test_metrics_gbdt = evaluate_count_forecast(
        test_pred_df[target_col].to_numpy(),
        test_pred_df["gbdt_prediction"].to_numpy(),
    )

    ll_base = aggregate_user_loglik(
        test_pred_df["user_id"].to_numpy(),
        test_pred_df[target_col].to_numpy(),
        test_pred_df["personalized_poisson_prediction"].to_numpy(),
        ll_col="ll_personalized_poisson",
    )
    ll_gbdt = aggregate_user_loglik(
        test_pred_df["user_id"].to_numpy(),
        test_pred_df[target_col].to_numpy(),
        test_pred_df["gbdt_prediction"].to_numpy(),
        ll_col="ll_experimental_gbdt",
    )
    test_purchases = (
        test_pred_df.groupby("user_id", as_index=False)[target_col]
        .sum()
        .rename(columns={target_col: "test_purchases"})
    )
    user_ll = ll_base.merge(ll_gbdt, on="user_id", how="outer").merge(test_purchases, on="user_id", how="left")
    user_ll = user_ll.fillna(0.0).sort_values("user_id").reset_index(drop=True)
    user_ll.to_csv(output_dir / "user_ll_scores.csv", index=False)

    daily_summary = (
        pred_df.groupby("event_date")[[target_col, "personalized_poisson_prediction", "gbdt_prediction"]]
        .mean()
        .reset_index()
    )
    daily_summary["split"] = np.where(daily_summary["event_date"] <= split.split_date, "train", "test")
    daily_summary.to_csv(output_dir / "daily_mean_summary.csv", index=False)

    plot_daily_aggregate_hawkes_vs_baseline(
        pred_df=pred_df,
        split_date=split.split_date,
        target_col=target_col,
        baseline_col="personalized_poisson_prediction",
        model_col="gbdt_prediction",
        out_path=output_dir / "daily_aggregate_analysis_window.png",
        baseline_label="Personalized Poisson",
        model_label="Experimental GBDT",
        title="Daily aggregate intensity: personalized Poisson vs GBDT",
    )
    plot_user_ll_gain_histogram(
        user_ll_df=user_ll,
        baseline_col="ll_personalized_poisson",
        model_col="ll_experimental_gbdt",
        out_path=output_dir / "user_ll_gain_hist.png",
        xlabel="Delta user-level test LL (GBDT - personalized Poisson)",
        title="User-level LL gain of experimental GBDT",
    )
    plot_delta_ll_vs_test_purchases(
        user_ll_df=user_ll,
        prev_col="ll_personalized_poisson",
        new_col="ll_experimental_gbdt",
        purchases_col="test_purchases",
        out_path=output_dir / "delta_ll_vs_test_purchases_personalized_to_gbdt.png",
        title="Per-user delta LL vs test purchases: personalized Poisson -> GBDT",
    )

    summary = {
        "data_path": str(Path(data_path)),
        "target_col": target_col,
        "analysis_window": {
            "start": str(analysis_start_ts.date()),
            "end": str(analysis_end_ts.date()),
        },
        "train_ratio": float(train_ratio),
        "split_date": str(split.split_date.date()),
        "source_features": list(source_features),
        "feature_count": int(len(feature_table.feature_names)),
        "model_params": {
            "seed": int(seed),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "max_iter": int(max_iter),
            "min_samples_leaf": int(min_samples_leaf),
            "max_users": int(max_users) if max_users is not None else None,
        },
        "train_panel": _panel_stats(split.train, target_col),
        "test_panel": _panel_stats(split.test, target_col),
        "train_metrics_personalized_poisson": train_metrics_base,
        "test_metrics_personalized_poisson": test_metrics_base,
        "train_metrics_gbdt": train_metrics_gbdt,
        "test_metrics_gbdt": test_metrics_gbdt,
        "test_improvement_vs_personalized_poisson": {
            "delta_poisson_loglik": float(test_metrics_gbdt["poisson_loglik"] - test_metrics_base["poisson_loglik"]),
            "delta_mean_poisson_nll": float(test_metrics_gbdt["mean_poisson_nll"] - test_metrics_base["mean_poisson_nll"]),
            "delta_mean_poisson_deviance": float(
                test_metrics_gbdt["mean_poisson_deviance"] - test_metrics_base["mean_poisson_deviance"]
            ),
            "delta_mae": float(test_metrics_gbdt["mae"] - test_metrics_base["mae"]),
            "delta_rmse": float(test_metrics_gbdt["rmse"] - test_metrics_base["rmse"]),
        },
        "user_level_ll": _pair_ll_summary(
            user_ll,
            prev_col="ll_personalized_poisson",
            new_col="ll_experimental_gbdt",
        ),
        "user_level_ll_by_test_purchases": _delta_by_test_purchase_bucket(
            user_ll,
            prev_col="ll_personalized_poisson",
            new_col="ll_experimental_gbdt",
        ),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

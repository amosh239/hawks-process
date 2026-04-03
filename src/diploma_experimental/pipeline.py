from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.diploma_baselines.data import filter_date_range, load_daily_grid, split_panel_by_date
from src.diploma_baselines.metrics import aggregate_user_loglik, evaluate_count_forecast
from src.diploma_baselines.models.personalized_gamma_poisson import PersonalizedGammaPoissonScaler
from src.diploma_baselines.models.rolling_seasonal_poisson import GlobalRollingSeasonalPoissonModel
from src.diploma_baselines.plots import plot_delta_ll_vs_test_purchases

from .hawkes import (
    DEFAULT_HALF_LIVES,
    FEATURE_NAMES,
    build_basis_states,
    fit_user_scales_with_fixed_excitation,
    fit_pooled_additive_multi_kernel_hawkes,
    predict_pooled_additive_multi_kernel_hawkes,
)
from .gbdt import SOURCE_FEATURES, build_feature_tables, fit_global_poisson_gbdt
from .plots import (
    plot_daily_aggregate_hawkes_vs_baseline,
    plot_hawkes_alpha_heatmap,
    plot_user_scale_histogram,
    plot_user_ll_gain_histogram,
)


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


def run_experimental_1_hawkes(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "to_ord",
    train_ratio: float = 0.8,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
    window_size: int = 7,
    half_lives: tuple[float, ...] = DEFAULT_HALF_LIVES,
    feature_names: list[str] | tuple[str, ...] | None = None,
    alpha_l2: float = 1e-4,
    learn_base_scale: bool = False,
    scale_l2: float = 0.0,
    scale_init: float = 1.0,
    max_iter: int = 120,
    model_label: str = "Experimental Hawkes",
    model_slug: str = "experimental_hawkes",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = tuple(feature_names or FEATURE_NAMES)
    value_cols = list(dict.fromkeys([target_col, *feature_names]))
    full_df = load_daily_grid(data_path, value_cols=value_cols)
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

    pred_df = analysis_df.copy()
    pred_df["personalized_poisson_prediction"] = scaler.predict(
        pred_df["user_id"].to_numpy(),
        global_base_analysis,
        method="posterior_mean",
    )

    beta = np.log(2.0) / np.asarray(half_lives, dtype=float)
    full_groups = full_df.groupby("user_id", sort=False)
    pred_groups = pred_df.groupby("user_id", sort=False)

    records: list[dict[str, object]] = []
    train_state_blocks: list[np.ndarray] = []
    train_y_blocks: list[np.ndarray] = []
    train_base_blocks: list[np.ndarray] = []

    split_date64 = np.datetime64(split.split_date)
    for (user_full_id, full_user), (user_pred_id, pred_user) in zip(full_groups, pred_groups):
        if user_full_id != user_pred_id:
            raise ValueError("User group alignment mismatch while preparing Hawkes records")

        x_full = full_user.loc[:, feature_names].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        full_dates = full_user["event_date"].to_numpy(dtype="datetime64[ns]")
        analysis_mask = (full_dates >= np.datetime64(analysis_start_ts)) & (full_dates <= np.datetime64(analysis_end_ts))
        states_analysis = states_full[analysis_mask]

        pred_dates = pred_user["event_date"].to_numpy(dtype="datetime64[ns]")
        train_mask = pred_dates <= split_date64
        y_analysis = pred_user[target_col].to_numpy(dtype=float)
        base_analysis = pred_user["personalized_poisson_prediction"].to_numpy(dtype=float)

        train_state_blocks.append(states_analysis[train_mask])
        train_y_blocks.append(y_analysis[train_mask])
        train_base_blocks.append(base_analysis[train_mask])

        records.append(
            {
                "frame": pred_user.loc[:, ["user_id", "event_date", target_col, "personalized_poisson_prediction"]].copy(),
                "states_analysis": states_analysis,
            }
        )

    hawkes = fit_pooled_additive_multi_kernel_hawkes(
        state_blocks=train_state_blocks,
        y_blocks=train_y_blocks,
        base_blocks=train_base_blocks,
        half_lives=tuple(float(x) for x in half_lives),
        feature_names=tuple(feature_names),
        alpha_l2=float(alpha_l2),
        learn_base_scale=bool(learn_base_scale),
        scale_l2=float(scale_l2),
        scale_init=float(scale_init),
        max_iter=int(max_iter),
    )

    hawkes_preds: list[np.ndarray] = []
    excitations: list[np.ndarray] = []
    for rec in records:
        frame = rec["frame"]
        states = np.asarray(rec["states_analysis"], dtype=float)
        lam, excitation = predict_pooled_additive_multi_kernel_hawkes(
            hawkes,
            states=states,
            base_lambda=frame["personalized_poisson_prediction"].to_numpy(dtype=float),
        )
        hawkes_preds.append(lam)
        excitations.append(excitation)

    pred_df["hawkes_prediction"] = np.concatenate(hawkes_preds)
    pred_df["hawkes_excitation"] = np.concatenate(excitations)
    pred_df = pred_df.drop(columns=["dow"])

    train_pred = pred_df[pred_df["event_date"] <= split.split_date]
    test_pred = pred_df[pred_df["event_date"] > split.split_date]

    train_metrics_base = evaluate_count_forecast(
        train_pred[target_col].to_numpy(),
        train_pred["personalized_poisson_prediction"].to_numpy(),
    )
    test_metrics_base = evaluate_count_forecast(
        test_pred[target_col].to_numpy(),
        test_pred["personalized_poisson_prediction"].to_numpy(),
    )
    train_metrics_hawkes = evaluate_count_forecast(
        train_pred[target_col].to_numpy(),
        train_pred["hawkes_prediction"].to_numpy(),
    )
    test_metrics_hawkes = evaluate_count_forecast(
        test_pred[target_col].to_numpy(),
        test_pred["hawkes_prediction"].to_numpy(),
    )

    ll_base = aggregate_user_loglik(
        test_pred["user_id"].to_numpy(),
        test_pred[target_col].to_numpy(),
        test_pred["personalized_poisson_prediction"].to_numpy(),
        ll_col="ll_personalized_poisson",
    )
    ll_hawkes = aggregate_user_loglik(
        test_pred["user_id"].to_numpy(),
        test_pred[target_col].to_numpy(),
        test_pred["hawkes_prediction"].to_numpy(),
        ll_col=f"ll_{model_slug}",
    )
    test_purchases = (
        test_pred.groupby("user_id", as_index=False)[target_col]
        .sum()
        .rename(columns={target_col: "test_purchases"})
    )
    user_ll = ll_base.merge(ll_hawkes, on="user_id", how="outer").merge(test_purchases, on="user_id", how="left")
    user_ll = user_ll.fillna(0.0).sort_values("user_id").reset_index(drop=True)
    user_ll.to_csv(output_dir / "user_ll_scores.csv", index=False)

    daily_summary = (
        pred_df.groupby("event_date")[[target_col, "personalized_poisson_prediction", "hawkes_prediction", "hawkes_excitation"]]
        .mean()
        .reset_index()
    )
    daily_summary["split"] = np.where(daily_summary["event_date"] <= split.split_date, "train", "test")
    daily_summary.to_csv(output_dir / "daily_mean_summary.csv", index=False)

    alpha_table = pd.DataFrame(
        hawkes.alpha_matrix(),
        index=list(feature_names),
        columns=[f"hl_{int(x) if float(x).is_integer() else x}" for x in half_lives],
    )
    alpha_table.to_csv(output_dir / "alpha_table.csv")

    plot_daily_aggregate_hawkes_vs_baseline(
        pred_df=pred_df,
        split_date=split.split_date,
        target_col=target_col,
        baseline_col="personalized_poisson_prediction",
        model_col="hawkes_prediction",
        out_path=output_dir / "daily_aggregate_analysis_window.png",
        baseline_label="Personalized Poisson",
        model_label=model_label,
        title=f"Daily aggregate intensity: personalized Poisson vs {model_label}",
    )
    plot_hawkes_alpha_heatmap(
        alpha_matrix=hawkes.alpha_matrix(),
        feature_names=feature_names,
        half_lives=half_lives,
        out_path=output_dir / "alpha_heatmap.png",
    )
    plot_user_ll_gain_histogram(
        user_ll_df=user_ll,
        baseline_col="ll_personalized_poisson",
        model_col=f"ll_{model_slug}",
        out_path=output_dir / "user_ll_gain_hist.png",
        xlabel=f"Delta user-level test LL ({model_label} - personalized Poisson)",
        title=f"User-level LL gain of {model_label}",
    )
    plot_delta_ll_vs_test_purchases(
        user_ll_df=user_ll,
        prev_col="ll_personalized_poisson",
        new_col=f"ll_{model_slug}",
        purchases_col="test_purchases",
        out_path=output_dir / "delta_ll_vs_test_purchases_personalized_to_hawkes.png",
        title=f"Per-user delta LL vs test purchases: personalized Poisson -> {model_label}",
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
        "feature_names": list(feature_names),
        "half_lives": [float(x) for x in half_lives],
        "alpha_l2": float(alpha_l2),
        "learn_base_scale": bool(learn_base_scale),
        "scale_l2": float(scale_l2),
        "scale_init": float(scale_init),
        "learned_base_scale": float(hawkes.base_scale),
        "max_iter": int(max_iter),
        "hawkes_fit_success": bool(hawkes.success),
        "train_panel": _panel_stats(split.train, target_col),
        "test_panel": _panel_stats(split.test, target_col),
        "train_metrics_personalized_poisson": train_metrics_base,
        "test_metrics_personalized_poisson": test_metrics_base,
        "train_metrics_hawkes": train_metrics_hawkes,
        "test_metrics_hawkes": test_metrics_hawkes,
        "test_improvement_vs_personalized_poisson": {
            "delta_poisson_loglik": float(test_metrics_hawkes["poisson_loglik"] - test_metrics_base["poisson_loglik"]),
            "delta_mean_poisson_nll": float(test_metrics_hawkes["mean_poisson_nll"] - test_metrics_base["mean_poisson_nll"]),
            "delta_mean_poisson_deviance": float(
                test_metrics_hawkes["mean_poisson_deviance"] - test_metrics_base["mean_poisson_deviance"]
            ),
            "delta_mae": float(test_metrics_hawkes["mae"] - test_metrics_base["mae"]),
            "delta_rmse": float(test_metrics_hawkes["rmse"] - test_metrics_base["rmse"]),
        },
        "user_level_ll": _pair_ll_summary(
            user_ll,
            prev_col="ll_personalized_poisson",
            new_col=f"ll_{model_slug}",
        ),
        "user_level_ll_by_test_purchases": _delta_by_test_purchase_bucket(
            user_ll,
            prev_col="ll_personalized_poisson",
            new_col=f"ll_{model_slug}",
        ),
        "alpha_matrix": {
            name: [float(x) for x in row]
            for name, row in zip(feature_names, hawkes.alpha_matrix())
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def run_experimental_1_1_hawkes(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "to_ord",
    train_ratio: float = 0.8,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
    window_size: int = 7,
    half_lives: tuple[float, ...] = DEFAULT_HALF_LIVES,
    feature_names: list[str] | tuple[str, ...] | None = None,
    alpha_l2: float = 1e-4,
    scale_l2: float = 10.0,
    scale_init: float = 1.0,
    user_scale_lower: float = 0.0,
    user_scale_upper: float = 5.0,
    user_scale_tol: float = 1e-8,
    user_scale_max_iter: int = 80,
    max_iter: int = 300,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = tuple(feature_names or FEATURE_NAMES)
    value_cols = list(dict.fromkeys([target_col, *feature_names]))
    full_df = load_daily_grid(data_path, value_cols=value_cols)
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

    pred_df = analysis_df.copy()
    pred_df["personalized_poisson_prediction"] = scaler.predict(
        pred_df["user_id"].to_numpy(),
        global_base_analysis,
        method="posterior_mean",
    )

    beta = np.log(2.0) / np.asarray(half_lives, dtype=float)
    full_groups = full_df.groupby("user_id", sort=False)
    pred_groups = pred_df.groupby("user_id", sort=False)

    records: list[dict[str, object]] = []
    train_state_blocks: list[np.ndarray] = []
    train_y_blocks: list[np.ndarray] = []
    train_base_blocks: list[np.ndarray] = []

    split_date64 = np.datetime64(split.split_date)
    for (user_full_id, full_user), (user_pred_id, pred_user) in zip(full_groups, pred_groups):
        if user_full_id != user_pred_id:
            raise ValueError("User group alignment mismatch while preparing Hawkes records")

        x_full = full_user.loc[:, feature_names].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        full_dates = full_user["event_date"].to_numpy(dtype="datetime64[ns]")
        analysis_mask = (full_dates >= np.datetime64(analysis_start_ts)) & (full_dates <= np.datetime64(analysis_end_ts))
        states_analysis = states_full[analysis_mask]

        pred_dates = pred_user["event_date"].to_numpy(dtype="datetime64[ns]")
        train_mask = pred_dates <= split_date64
        y_analysis = pred_user[target_col].to_numpy(dtype=float)
        base_analysis = pred_user["personalized_poisson_prediction"].to_numpy(dtype=float)

        train_state_blocks.append(states_analysis[train_mask])
        train_y_blocks.append(y_analysis[train_mask])
        train_base_blocks.append(base_analysis[train_mask])

        records.append(
            {
                "user_id": user_pred_id,
                "frame": pred_user.loc[:, ["user_id", "event_date", target_col, "personalized_poisson_prediction"]].copy(),
                "states_analysis": states_analysis,
                "train_mask": train_mask,
            }
        )

    global_hawkes = fit_pooled_additive_multi_kernel_hawkes(
        state_blocks=train_state_blocks,
        y_blocks=train_y_blocks,
        base_blocks=train_base_blocks,
        half_lives=tuple(float(x) for x in half_lives),
        feature_names=tuple(feature_names),
        alpha_l2=float(alpha_l2),
        learn_base_scale=True,
        scale_l2=float(scale_l2),
        scale_init=float(scale_init),
        max_iter=int(max_iter),
    )

    train_excitation_blocks: list[np.ndarray] = []
    user_frames: list[pd.DataFrame] = []
    for rec in records:
        frame = rec["frame"]
        states = np.asarray(rec["states_analysis"], dtype=float)
        global_lam, excitation = predict_pooled_additive_multi_kernel_hawkes(
            global_hawkes,
            states=states,
            base_lambda=frame["personalized_poisson_prediction"].to_numpy(dtype=float),
        )
        train_mask = np.asarray(rec["train_mask"], dtype=bool)
        train_excitation_blocks.append(excitation[train_mask])

        frame = frame.copy()
        frame["global_scaled_hawkes_prediction"] = global_lam
        frame["hawkes_excitation"] = excitation
        user_frames.append(frame)

    user_scale_fit = fit_user_scales_with_fixed_excitation(
        y_blocks=train_y_blocks,
        base_blocks=train_base_blocks,
        excitation_blocks=train_excitation_blocks,
        init_scale=float(global_hawkes.base_scale),
        lower=float(user_scale_lower),
        upper=float(user_scale_upper),
        tol=float(user_scale_tol),
        max_iter=int(user_scale_max_iter),
    )

    scale_rows: list[dict[str, float | int]] = []
    personalized_hawkes_frames: list[pd.DataFrame] = []
    for frame, rec, user_scale in zip(user_frames, records, user_scale_fit.scales):
        frame = frame.copy()
        frame["user_specific_scale"] = float(user_scale)
        frame["user_scaled_hawkes_prediction"] = (
            float(user_scale) * frame["personalized_poisson_prediction"].to_numpy(dtype=float)
            + frame["hawkes_excitation"].to_numpy(dtype=float)
        )
        personalized_hawkes_frames.append(frame)
        scale_rows.append(
            {
                "user_id": int(rec["user_id"]),
                "user_scale": float(user_scale),
                "train_days": int(np.sum(np.asarray(rec["train_mask"], dtype=bool))),
            }
        )

    pred_df = pd.concat(personalized_hawkes_frames, ignore_index=True).sort_values(["user_id", "event_date"]).reset_index(drop=True)
    pd.DataFrame(scale_rows).sort_values("user_id").to_csv(output_dir / "user_scales.csv", index=False)

    train_pred = pred_df[pred_df["event_date"] <= split.split_date]
    test_pred = pred_df[pred_df["event_date"] > split.split_date]

    train_metrics_base = evaluate_count_forecast(
        train_pred[target_col].to_numpy(),
        train_pred["personalized_poisson_prediction"].to_numpy(),
    )
    test_metrics_base = evaluate_count_forecast(
        test_pred[target_col].to_numpy(),
        test_pred["personalized_poisson_prediction"].to_numpy(),
    )
    train_metrics_e11 = evaluate_count_forecast(
        train_pred[target_col].to_numpy(),
        train_pred["global_scaled_hawkes_prediction"].to_numpy(),
    )
    test_metrics_e11 = evaluate_count_forecast(
        test_pred[target_col].to_numpy(),
        test_pred["global_scaled_hawkes_prediction"].to_numpy(),
    )
    train_metrics_e12 = evaluate_count_forecast(
        train_pred[target_col].to_numpy(),
        train_pred["user_scaled_hawkes_prediction"].to_numpy(),
    )
    test_metrics_e12 = evaluate_count_forecast(
        test_pred[target_col].to_numpy(),
        test_pred["user_scaled_hawkes_prediction"].to_numpy(),
    )

    ll_base = aggregate_user_loglik(
        test_pred["user_id"].to_numpy(),
        test_pred[target_col].to_numpy(),
        test_pred["personalized_poisson_prediction"].to_numpy(),
        ll_col="ll_personalized_poisson",
    )
    ll_e11 = aggregate_user_loglik(
        test_pred["user_id"].to_numpy(),
        test_pred[target_col].to_numpy(),
        test_pred["global_scaled_hawkes_prediction"].to_numpy(),
        ll_col="ll_global_scaled_hawkes",
    )
    ll_e12 = aggregate_user_loglik(
        test_pred["user_id"].to_numpy(),
        test_pred[target_col].to_numpy(),
        test_pred["user_scaled_hawkes_prediction"].to_numpy(),
        ll_col="ll_user_scale_hawkes",
    )
    test_purchases = (
        test_pred.groupby("user_id", as_index=False)[target_col]
        .sum()
        .rename(columns={target_col: "test_purchases"})
    )
    user_ll = (
        ll_base.merge(ll_e11, on="user_id", how="outer")
        .merge(ll_e12, on="user_id", how="outer")
        .merge(test_purchases, on="user_id", how="left")
    )
    user_ll = user_ll.fillna(0.0).sort_values("user_id").reset_index(drop=True)
    user_ll.to_csv(output_dir / "user_ll_scores.csv", index=False)

    daily_summary = (
        pred_df.groupby("event_date")[
            [
                target_col,
                "personalized_poisson_prediction",
                "global_scaled_hawkes_prediction",
                "user_scaled_hawkes_prediction",
                "hawkes_excitation",
            ]
        ]
        .mean()
        .reset_index()
    )
    daily_summary["split"] = np.where(daily_summary["event_date"] <= split.split_date, "train", "test")
    daily_summary.to_csv(output_dir / "daily_mean_summary.csv", index=False)

    alpha_table = pd.DataFrame(
        global_hawkes.alpha_matrix(),
        index=list(feature_names),
        columns=[f"hl_{int(x) if float(x).is_integer() else x}" for x in half_lives],
    )
    alpha_table.to_csv(output_dir / "alpha_table.csv")

    plot_daily_aggregate_hawkes_vs_baseline(
        pred_df=pred_df,
        split_date=split.split_date,
        target_col=target_col,
        baseline_col="personalized_poisson_prediction",
        model_col="user_scaled_hawkes_prediction",
        out_path=output_dir / "daily_aggregate_analysis_window.png",
        baseline_label="Personalized Poisson",
        model_label="User-scale Hawkes",
        title="Daily aggregate intensity: personalized Poisson vs user-scale Hawkes",
    )
    plot_hawkes_alpha_heatmap(
        alpha_matrix=global_hawkes.alpha_matrix(),
        feature_names=feature_names,
        half_lives=half_lives,
        out_path=output_dir / "alpha_heatmap.png",
    )
    plot_user_ll_gain_histogram(
        user_ll_df=user_ll,
        baseline_col="ll_personalized_poisson",
        model_col="ll_user_scale_hawkes",
        out_path=output_dir / "user_ll_gain_hist.png",
        xlabel="Delta user-level test LL (user-scale Hawkes - personalized Poisson)",
        title="User-level LL gain of user-scale Hawkes",
    )
    plot_delta_ll_vs_test_purchases(
        user_ll_df=user_ll,
        prev_col="ll_personalized_poisson",
        new_col="ll_user_scale_hawkes",
        purchases_col="test_purchases",
        out_path=output_dir / "delta_ll_vs_test_purchases_personalized_to_hawkes.png",
        title="Per-user delta LL vs test purchases: personalized Poisson -> user-scale Hawkes",
    )
    plot_user_scale_histogram(
        scales=user_scale_fit.scales,
        out_path=output_dir / "user_scale_hist.png",
        title="Distribution of fitted user-specific baseline scales",
        xlabel="Fitted user-specific scale c_u",
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
        "feature_names": list(feature_names),
        "half_lives": [float(x) for x in half_lives],
        "alpha_l2": float(alpha_l2),
        "scale_l2": float(scale_l2),
        "scale_init": float(scale_init),
        "global_hawkes_fit_success": bool(global_hawkes.success),
        "global_base_scale": float(global_hawkes.base_scale),
        "user_scale_fit": {
            "lower": float(user_scale_lower),
            "upper": float(user_scale_upper),
            "tol": float(user_scale_tol),
            "max_iter": int(user_scale_max_iter),
            "mean_scale": float(user_scale_fit.mean_scale),
            "median_scale": float(user_scale_fit.median_scale),
            "lower_bound_hits": int(user_scale_fit.lower_bound_hits),
            "upper_bound_hits": int(user_scale_fit.upper_bound_hits),
        },
        "train_panel": _panel_stats(split.train, target_col),
        "test_panel": _panel_stats(split.test, target_col),
        "train_metrics_personalized_poisson": train_metrics_base,
        "test_metrics_personalized_poisson": test_metrics_base,
        "train_metrics_global_scaled_hawkes": train_metrics_e11,
        "test_metrics_global_scaled_hawkes": test_metrics_e11,
        "train_metrics_user_scale_hawkes": train_metrics_e12,
        "test_metrics_user_scale_hawkes": test_metrics_e12,
        "test_improvement_vs_personalized_poisson": {
            "delta_poisson_loglik": float(test_metrics_e12["poisson_loglik"] - test_metrics_base["poisson_loglik"]),
            "delta_mean_poisson_nll": float(test_metrics_e12["mean_poisson_nll"] - test_metrics_base["mean_poisson_nll"]),
            "delta_mean_poisson_deviance": float(
                test_metrics_e12["mean_poisson_deviance"] - test_metrics_base["mean_poisson_deviance"]
            ),
            "delta_mae": float(test_metrics_e12["mae"] - test_metrics_base["mae"]),
            "delta_rmse": float(test_metrics_e12["rmse"] - test_metrics_base["rmse"]),
        },
        "test_improvement_vs_global_scaled_hawkes": {
            "delta_poisson_loglik": float(test_metrics_e12["poisson_loglik"] - test_metrics_e11["poisson_loglik"]),
            "delta_mean_poisson_nll": float(test_metrics_e12["mean_poisson_nll"] - test_metrics_e11["mean_poisson_nll"]),
            "delta_mean_poisson_deviance": float(
                test_metrics_e12["mean_poisson_deviance"] - test_metrics_e11["mean_poisson_deviance"]
            ),
            "delta_mae": float(test_metrics_e12["mae"] - test_metrics_e11["mae"]),
            "delta_rmse": float(test_metrics_e12["rmse"] - test_metrics_e11["rmse"]),
        },
        "user_level_ll_vs_personalized_poisson": _pair_ll_summary(
            user_ll,
            prev_col="ll_personalized_poisson",
            new_col="ll_user_scale_hawkes",
        ),
        "user_level_ll_by_test_purchases_vs_personalized_poisson": _delta_by_test_purchase_bucket(
            user_ll,
            prev_col="ll_personalized_poisson",
            new_col="ll_user_scale_hawkes",
        ),
        "user_level_ll_vs_global_scaled_hawkes": _pair_ll_summary(
            user_ll,
            prev_col="ll_global_scaled_hawkes",
            new_col="ll_user_scale_hawkes",
        ),
        "user_level_ll_by_test_purchases_vs_global_scaled_hawkes": _delta_by_test_purchase_bucket(
            user_ll,
            prev_col="ll_global_scaled_hawkes",
            new_col="ll_user_scale_hawkes",
        ),
        "alpha_matrix": {
            name: [float(x) for x in row]
            for name, row in zip(feature_names, global_hawkes.alpha_matrix())
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


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

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import filter_date_range, load_daily_grid, split_panel_by_date
from .experiment_utils import _resolve_analysis_window
from .metrics import evaluate_count_forecast
from .models.hawkes import (
    FEATURE_NAMES,
    PooledAdditiveMultiKernelHawkesResult,
    build_basis_states,
    fit_pooled_additive_multi_kernel_hawkes,
    predict_pooled_additive_multi_kernel_hawkes,
)
from .models.personalized_gamma_poisson import PersonalizedGammaPoissonScaler
from .models.rolling_seasonal_poisson import GlobalRollingSeasonalPoissonModel


@dataclass
class HawkesUserRecord:
    user_id: int
    event_date: np.ndarray
    y: np.ndarray
    base_lambda: np.ndarray
    states: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray


def _relative_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def _prepare_records(
    data_path: str | Path,
    target_col: str,
    train_ratio: float,
    analysis_start: str | None,
    analysis_end: str | None,
    window_size: int,
    half_lives: tuple[float, ...],
    feature_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, list[HawkesUserRecord]]:
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

    pred_df = analysis_df.loc[:, ["user_id", "event_date", target_col]].copy()
    pred_df["personalized_poisson_prediction"] = scaler.predict(
        pred_df["user_id"].to_numpy(),
        global_base_analysis,
        method="posterior_mean",
    )

    beta = np.log(2.0) / np.asarray(half_lives, dtype=float)
    full_groups = full_df.groupby("user_id", sort=False)
    pred_groups = pred_df.groupby("user_id", sort=False)
    split_date64 = np.datetime64(split.split_date)

    records: list[HawkesUserRecord] = []
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
        test_mask = pred_dates > split_date64

        records.append(
            HawkesUserRecord(
                user_id=int(user_pred_id),
                event_date=pred_dates,
                y=pred_user[target_col].to_numpy(dtype=float),
                base_lambda=pred_user["personalized_poisson_prediction"].to_numpy(dtype=float),
                states=states_analysis,
                train_mask=train_mask,
                test_mask=test_mask,
            )
        )

    return full_df, pred_df, split.split_date, records


def _fit_hawkes_on_subset(
    records: list[HawkesUserRecord],
    selected_train_dates: pd.Index,
    half_lives: tuple[float, ...],
    feature_names: tuple[str, ...],
    alpha_l2: float,
    scale_l2: float,
    max_iter: int,
    ref_model: PooledAdditiveMultiKernelHawkesResult | None = None,
) -> PooledAdditiveMultiKernelHawkesResult:
    selected_dates64 = selected_train_dates.to_numpy(dtype="datetime64[ns]")
    state_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    base_blocks: list[np.ndarray] = []

    for rec in records:
        use_mask = np.asarray(rec.train_mask & np.isin(rec.event_date, selected_dates64), dtype=bool)
        if not np.any(use_mask):
            continue
        state_blocks.append(rec.states[use_mask])
        y_blocks.append(rec.y[use_mask])
        base_blocks.append(rec.base_lambda[use_mask])

    return fit_pooled_additive_multi_kernel_hawkes(
        state_blocks=state_blocks,
        y_blocks=y_blocks,
        base_blocks=base_blocks,
        half_lives=half_lives,
        feature_names=tuple(feature_names),
        alpha_l2=float(alpha_l2),
        learn_base_scale=True,
        scale_l2=float(scale_l2),
        scale_init=float(ref_model.base_scale if ref_model is not None else 1.0),
        alpha_init=None if ref_model is None else np.asarray(ref_model.alpha, dtype=float),
        max_iter=int(max_iter),
    )


def _evaluate_model_on_test(
    model: PooledAdditiveMultiKernelHawkesResult,
    records: list[HawkesUserRecord],
) -> dict[str, float]:
    y_blocks: list[np.ndarray] = []
    base_blocks: list[np.ndarray] = []
    model_blocks: list[np.ndarray] = []
    for rec in records:
        if not np.any(rec.test_mask):
            continue
        y_blocks.append(rec.y[rec.test_mask])
        base_blocks.append(rec.base_lambda[rec.test_mask])
        lam, _ = predict_pooled_additive_multi_kernel_hawkes(
            model,
            states=rec.states[rec.test_mask],
            base_lambda=rec.base_lambda[rec.test_mask],
        )
        model_blocks.append(lam)

    y = np.concatenate(y_blocks)
    base = np.concatenate(base_blocks)
    pred = np.concatenate(model_blocks)
    base_metrics = evaluate_count_forecast(y, base)
    model_metrics = evaluate_count_forecast(y, pred)
    return {
        "test_poisson_loglik": float(model_metrics["poisson_loglik"]),
        "test_delta_poisson_loglik_vs_personalized": float(model_metrics["poisson_loglik"] - base_metrics["poisson_loglik"]),
        "test_mean_poisson_nll": float(model_metrics["mean_poisson_nll"]),
        "test_delta_mean_poisson_nll_vs_personalized": float(model_metrics["mean_poisson_nll"] - base_metrics["mean_poisson_nll"]),
        "test_mae": float(model_metrics["mae"]),
        "test_delta_mae_vs_personalized": float(model_metrics["mae"] - base_metrics["mae"]),
    }


def _collect_test_excitation_frame(model, records) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for rec in records:
        if not np.any(rec.test_mask):
            continue
        lam, excitation = predict_pooled_additive_multi_kernel_hawkes(
            model,
            states=rec.states[rec.test_mask],
            base_lambda=rec.base_lambda[rec.test_mask],
        )
        rows.append(
            pd.DataFrame(
                {
                    "user_id": int(rec.user_id),
                    "event_date": pd.to_datetime(rec.event_date[rec.test_mask]),
                    "y": rec.y[rec.test_mask].astype(float),
                    "base_lambda": rec.base_lambda[rec.test_mask].astype(float),
                    "excitation": excitation.astype(float),
                    "prediction": lam.astype(float),
                }
            )
        )
    if not rows:
        raise ValueError("No test rows available for excitation research")
    return pd.concat(rows, ignore_index=True).sort_values(["user_id", "event_date"]).reset_index(drop=True)


def _group_mass_rows(
    model,
    label: str,
    feature_names: tuple[str, ...],
) -> tuple[dict[str, float | str], dict[str, float | str]]:
    alpha_matrix = model.alpha_matrix()
    feature_row: dict[str, float | str] = {"label": label}
    for feature, value in zip(feature_names, alpha_matrix.sum(axis=1)):
        feature_row[feature] = float(value)

    half_life_row: dict[str, float | str] = {"label": label}
    for hl, value in zip(model.half_lives, alpha_matrix.sum(axis=0)):
        hl_label = str(int(hl)) if float(hl).is_integer() else str(float(hl))
        half_life_row[f"hl_{hl_label}"] = float(value)
    return feature_row, half_life_row


def _plot_fraction_similarity(summary_df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    df = summary_df.sort_values("train_fraction").copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    ax.plot(df["train_fraction"], df["alpha_relative_l2_to_full"], marker="o", color="#0B3C5D", label="alpha rel-L2")
    ax.plot(df["train_fraction"], df["feature_mass_relative_l2_to_full"], marker="s", color="#2E8B57", label="feature-mass rel-L2")
    ax.plot(df["train_fraction"], df["daily_excitation_relative_l2_to_full"], marker="^", color="#D2691E", label="daily excitation rel-L2")
    ax.plot(df["train_fraction"], df["userday_excitation_relative_l2_to_full"], marker="d", color="#6A994E", label="user-day excitation rel-L2")
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("Relative L2 to full fit")
    ax.set_title("Distance to full fit")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(df["train_fraction"], df["test_delta_poisson_loglik_vs_personalized"], marker="o", color="#0B3C5D", label="test delta loglik")
    ax.plot(df["train_fraction"], df["base_scale"], marker="s", color="#2E8B57", label="base scale c")
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("Metric value")
    ax.set_title("Test quality and learned scale")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_weekly_similarity(summary_df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    df = summary_df.sort_values("window_start").copy()
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    ax = axes[0]
    ax.plot(x, df["alpha_relative_l2_to_full"], marker="o", color="#0B3C5D", label="alpha rel-L2")
    ax.plot(x, df["feature_mass_relative_l2_to_full"], marker="s", color="#2E8B57", label="feature-mass rel-L2")
    ax.plot(x, df["daily_excitation_relative_l2_to_full"], marker="^", color="#D2691E", label="daily excitation rel-L2")
    ax.plot(x, df["userday_excitation_relative_l2_to_full"], marker="d", color="#6A994E", label="user-day excitation rel-L2")
    ax.set_xlabel("Train week index")
    ax.set_ylabel("Relative L2 to full fit")
    ax.set_title("Distance to full fit")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(x, df["test_delta_poisson_loglik_vs_personalized"], marker="o", color="#0B3C5D", label="test delta loglik")
    ax.plot(x, df["base_scale"], marker="s", color="#2E8B57", label="base scale c")
    ax.set_xlabel("Train week index")
    ax.set_ylabel("Metric value")
    ax.set_title("Test quality and learned scale")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fraction_daily_excitation_paths(paths_df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    df = paths_df.sort_values("event_date").copy()
    fig, ax = plt.subplots(figsize=(11.2, 4.8))

    for col, label, color, width in [
        ("full_fit", "Full fit", "#0B3C5D", 1.7),
        ("frac_0_1", "10% train", "#D2691E", 1.1),
        ("frac_0_3", "30% train", "#B56576", 1.1),
        ("frac_0_7", "70% train", "#2E8B57", 1.1),
    ]:
        if col in df.columns:
            ax.plot(df["event_date"], df[col], label=label, color=color, linewidth=width)

    ax.set_xlabel("Test date")
    ax.set_ylabel("Mean Hawkes excitation per user-day")
    ax.set_title("Daily aggregate excitation on test for different train fractions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_feature_mass_heatmap(df: pd.DataFrame, out_path: str | Path, title: str, x_label: str) -> None:
    out_path = Path(out_path)
    data = df.set_index("label").T
    values = data.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(list(data.columns), rotation=45, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(list(data.index))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Summed alpha over half-lives")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_feature_mass_heatmap_dual_scale(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str,
    x_label: str,
    clip_quantile: float = 0.95,
) -> None:
    out_path = Path(out_path)
    data = df.set_index("label").T
    values = data.to_numpy(dtype=float)
    clipped_vmax = float(np.quantile(values, clip_quantile))
    clipped_vmax = max(clipped_vmax, 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))

    ax = axes[0]
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(list(data.columns), rotation=45, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(list(data.index))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Feature")
    ax.set_title("Full scale")
    fig.colorbar(im, ax=ax, label="Summed alpha over half-lives")

    ax = axes[1]
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=clipped_vmax)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(list(data.columns), rotation=45, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(list(data.index))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Feature")
    ax.set_title(f"Clipped at q{int(clip_quantile * 100)}")
    fig.colorbar(im, ax=ax, label="Summed alpha over half-lives")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_hawkes_excitation_research(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "to_ord",
    train_ratio: float = 0.8,
    analysis_start: str | None = "2025-01-15",
    analysis_end: str | None = "2025-09-30",
    window_size: int = 7,
    half_lives: tuple[float, ...] = (1.0, 3.0),
    feature_names: list[str] | tuple[str, ...] | None = None,
    alpha_l2: float = 1e-4,
    scale_l2: float = 10.0,
    reference_max_iter: int = 300,
    subset_max_iter: int = 150,
    train_fractions: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0),
    weekly_window_days: int = 7,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = tuple(feature_names or FEATURE_NAMES)

    _, pred_df, split_date, records = _prepare_records(
        data_path=data_path,
        target_col=target_col,
        train_ratio=train_ratio,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        window_size=window_size,
        half_lives=half_lives,
        feature_names=feature_names,
    )

    train_dates = pd.Index(sorted(pd.to_datetime(pred_df.loc[pred_df["event_date"] <= split_date, "event_date"].unique())))
    if train_dates.empty:
        raise ValueError("No train dates available for excitation research")

    reference_model = _fit_hawkes_on_subset(
        records=records,
        selected_train_dates=train_dates,
        half_lives=half_lives,
        feature_names=feature_names,
        alpha_l2=alpha_l2,
        scale_l2=scale_l2,
        max_iter=reference_max_iter,
        ref_model=None,
    )
    reference_eval = _evaluate_model_on_test(reference_model, records)
    reference_frame = _collect_test_excitation_frame(reference_model, records)
    reference_daily = reference_frame.groupby("event_date", as_index=False)[["excitation", "prediction"]].mean()

    reference_feature_mass_row, reference_half_life_mass_row = _group_mass_rows(reference_model, "full_fit", feature_names)
    reference_feature_mass = np.array([reference_feature_mass_row[name] for name in feature_names], dtype=float)
    reference_half_life_labels = [key for key in reference_half_life_mass_row.keys() if key != "label"]
    reference_half_life_mass = np.array([reference_half_life_mass_row[key] for key in reference_half_life_labels], dtype=float)

    fraction_rows: list[dict[str, float | int | str]] = []
    fraction_feature_rows: list[dict[str, float | str]] = []
    fraction_half_life_rows: list[dict[str, float | str]] = []
    fraction_daily_paths = reference_daily.rename(columns={"excitation": "full_fit"}).loc[:, ["event_date", "full_fit"]]

    weekly_rows: list[dict[str, float | int | str]] = []
    weekly_feature_rows: list[dict[str, float | str]] = []
    weekly_half_life_rows: list[dict[str, float | str]] = []

    for frac in train_fractions:
        frac = float(frac)
        n_keep = max(1, min(len(train_dates), int(np.ceil(len(train_dates) * frac))))
        cutoff = train_dates[n_keep - 1]
        selected_dates = train_dates[train_dates <= cutoff]
        model = _fit_hawkes_on_subset(
            records=records,
            selected_train_dates=selected_dates,
            half_lives=half_lives,
            feature_names=feature_names,
            alpha_l2=alpha_l2,
            scale_l2=scale_l2,
            max_iter=subset_max_iter,
            ref_model=reference_model,
        )
        eval_metrics = _evaluate_model_on_test(model, records)
        test_frame = _collect_test_excitation_frame(model, records)
        test_daily = test_frame.groupby("event_date", as_index=False)[["excitation", "prediction"]].mean()

        feature_row, hl_row = _group_mass_rows(model, f"{frac:.1f}", feature_names)
        feature_mass = np.array([feature_row[name] for name in feature_names], dtype=float)
        half_life_mass = np.array([hl_row[key] for key in reference_half_life_labels], dtype=float)

        fraction_rows.append(
            {
                "train_fraction": frac,
                "train_days_used": int(n_keep),
                "window_end": str(pd.Timestamp(cutoff).date()),
                "fit_success": int(bool(model.success)),
                "base_scale": float(model.base_scale),
                "alpha_relative_l2_to_full": _relative_l2_distance(model.alpha, reference_model.alpha),
                "feature_mass_relative_l2_to_full": _relative_l2_distance(feature_mass, reference_feature_mass),
                "half_life_mass_relative_l2_to_full": _relative_l2_distance(half_life_mass, reference_half_life_mass),
                "userday_excitation_relative_l2_to_full": _relative_l2_distance(
                    test_frame["excitation"].to_numpy(),
                    reference_frame["excitation"].to_numpy(),
                ),
                "daily_excitation_relative_l2_to_full": _relative_l2_distance(
                    test_daily["excitation"].to_numpy(),
                    reference_daily["excitation"].to_numpy(),
                ),
                **eval_metrics,
            }
        )
        fraction_feature_rows.append(feature_row)
        fraction_half_life_rows.append(hl_row)

        frac_label = f"frac_{str(frac).replace('.', '_')}"
        if frac in {0.1, 0.3, 0.7}:
            fraction_daily_paths = fraction_daily_paths.merge(
                test_daily.rename(columns={"excitation": frac_label}).loc[:, ["event_date", frac_label]],
                on="event_date",
                how="left",
            )

    n_train_dates = len(train_dates)
    for start_idx in range(0, n_train_dates, int(weekly_window_days)):
        end_idx = min(n_train_dates, start_idx + int(weekly_window_days))
        if end_idx - start_idx < int(weekly_window_days):
            break
        start_date = train_dates[start_idx]
        end_date = train_dates[end_idx - 1]
        selected_dates = train_dates[(train_dates >= start_date) & (train_dates <= end_date)]
        model = _fit_hawkes_on_subset(
            records=records,
            selected_train_dates=selected_dates,
            half_lives=half_lives,
            feature_names=feature_names,
            alpha_l2=alpha_l2,
            scale_l2=scale_l2,
            max_iter=subset_max_iter,
            ref_model=reference_model,
        )
        eval_metrics = _evaluate_model_on_test(model, records)
        test_frame = _collect_test_excitation_frame(model, records)
        test_daily = test_frame.groupby("event_date", as_index=False)[["excitation", "prediction"]].mean()

        label = f"{pd.Timestamp(start_date).strftime('%m-%d')}..{pd.Timestamp(end_date).strftime('%m-%d')}"
        feature_row, hl_row = _group_mass_rows(model, label, feature_names)
        feature_mass = np.array([feature_row[name] for name in feature_names], dtype=float)
        half_life_mass = np.array([hl_row[key] for key in reference_half_life_labels], dtype=float)

        weekly_rows.append(
            {
                "window_index": int(start_idx // int(weekly_window_days)),
                "window_start": str(pd.Timestamp(start_date).date()),
                "window_end": str(pd.Timestamp(end_date).date()),
                "label": label,
                "fit_success": int(bool(model.success)),
                "base_scale": float(model.base_scale),
                "alpha_relative_l2_to_full": _relative_l2_distance(model.alpha, reference_model.alpha),
                "feature_mass_relative_l2_to_full": _relative_l2_distance(feature_mass, reference_feature_mass),
                "half_life_mass_relative_l2_to_full": _relative_l2_distance(half_life_mass, reference_half_life_mass),
                "userday_excitation_relative_l2_to_full": _relative_l2_distance(
                    test_frame["excitation"].to_numpy(),
                    reference_frame["excitation"].to_numpy(),
                ),
                "daily_excitation_relative_l2_to_full": _relative_l2_distance(
                    test_daily["excitation"].to_numpy(),
                    reference_daily["excitation"].to_numpy(),
                ),
                **eval_metrics,
            }
        )
        weekly_feature_rows.append(feature_row)
        weekly_half_life_rows.append(hl_row)

    fraction_summary_df = pd.DataFrame(fraction_rows).sort_values("train_fraction").reset_index(drop=True)
    fraction_feature_df = pd.DataFrame(fraction_feature_rows)
    fraction_half_life_df = pd.DataFrame(fraction_half_life_rows)
    weekly_summary_df = pd.DataFrame(weekly_rows).sort_values("window_start").reset_index(drop=True)
    weekly_feature_df = pd.DataFrame(weekly_feature_rows)
    weekly_half_life_df = pd.DataFrame(weekly_half_life_rows)

    fraction_summary_df.to_csv(output_dir / "fraction_excitation_summary.csv", index=False)
    fraction_feature_df.to_csv(output_dir / "fraction_feature_mass.csv", index=False)
    fraction_half_life_df.to_csv(output_dir / "fraction_half_life_mass.csv", index=False)
    fraction_daily_paths.to_csv(output_dir / "fraction_daily_excitation_paths.csv", index=False)
    weekly_summary_df.to_csv(output_dir / "weekly_excitation_summary.csv", index=False)
    weekly_feature_df.to_csv(output_dir / "weekly_feature_mass.csv", index=False)
    weekly_half_life_df.to_csv(output_dir / "weekly_half_life_mass.csv", index=False)

    _plot_fraction_similarity(fraction_summary_df, output_dir / "fraction_excitation_similarity.png")
    _plot_weekly_similarity(weekly_summary_df, output_dir / "weekly_excitation_similarity.png")
    _plot_fraction_daily_excitation_paths(fraction_daily_paths, output_dir / "fraction_daily_excitation_paths.png")
    _plot_feature_mass_heatmap(
        fraction_feature_df,
        output_dir / "fraction_feature_mass_heatmap.png",
        title="Feature-mass stability across train fractions",
        x_label="Train fraction",
    )
    _plot_feature_mass_heatmap_dual_scale(
        weekly_feature_df,
        output_dir / "weekly_feature_mass_heatmap.png",
        title="Feature-mass stability across weekly fits",
        x_label="Train week",
    )

    summary = {
        "data_path": str(Path(data_path)),
        "target_col": target_col,
        "feature_names": list(feature_names),
        "half_lives": [float(x) for x in half_lives],
        "analysis_window": {
            "start": str(pd.Timestamp(pred_df["event_date"].min()).date()),
            "end": str(pd.Timestamp(pred_df["event_date"].max()).date()),
        },
        "split_date": str(pd.Timestamp(split_date).date()),
        "reference_model": {
            "base_scale": float(reference_model.base_scale),
            "test_delta_poisson_loglik_vs_personalized": float(reference_eval["test_delta_poisson_loglik_vs_personalized"]),
        },
        "fraction_summary_csv": str(output_dir / "fraction_excitation_summary.csv"),
        "weekly_summary_csv": str(output_dir / "weekly_excitation_summary.csv"),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

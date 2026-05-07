from __future__ import annotations

import numpy as np
import pandas as pd


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

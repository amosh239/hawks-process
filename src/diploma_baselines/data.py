from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DAYUSES_ACTIVITY_COLS = [
    "search",
    "cat",
    "searches",
    "has_search_to_cart",
    "has_search_to_ord",
    "has_cat_to_cart",
    "has_cat_to_ord",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "to_cart",
    "to_ord",
    "gmv",
]


@dataclass(frozen=True)
class PanelSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    split_date: pd.Timestamp


def load_daily_grid(path: str | Path, value_cols: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    path = Path(path)
    value_cols = list(value_cols or [])
    usecols = ["user_id", "event_date", *value_cols]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["event_date"])
    df = df.sort_values(["user_id", "event_date"]).reset_index(drop=True)
    df["dow"] = df["event_date"].dt.dayofweek.astype(int)
    return df


def split_panel_by_date(df: pd.DataFrame, train_ratio: float = 0.8) -> PanelSplit:
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")

    unique_dates = pd.Index(sorted(df["event_date"].drop_duplicates()))
    if len(unique_dates) < 2:
        raise ValueError("Need at least two distinct dates for train/test split")

    split_idx = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * train_ratio)))
    split_date = unique_dates[split_idx - 1]

    train = df[df["event_date"] <= split_date].copy()
    test = df[df["event_date"] > split_date].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty partition")

    return PanelSplit(train=train, test=test, split_date=split_date)


def filter_date_range(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = df
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        out = out[out["event_date"] >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        out = out[out["event_date"] <= end_ts]
    return out.copy()

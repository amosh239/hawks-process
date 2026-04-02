from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class GlobalRollingMeanPoissonModel:
    window_size: int = 7
    min_periods: int = 1
    daily_prediction_: pd.Series | None = None

    def fit(self, daily_mean_series: pd.Series) -> "GlobalRollingMeanPoissonModel":
        if int(self.window_size) <= 0:
            raise ValueError("window_size must be positive")
        daily_mean_series = daily_mean_series.sort_index().astype(float)
        if daily_mean_series.empty:
            raise ValueError("Cannot fit rolling mean model on empty daily series")
        self.daily_prediction_ = daily_mean_series.shift(1).rolling(
            window=int(self.window_size),
            min_periods=int(self.min_periods),
        ).mean()
        return self

    def predict_for_dates(self, dates) -> pd.Series:
        if self.daily_prediction_ is None:
            raise ValueError("Model is not fitted")
        idx = pd.to_datetime(pd.Index(dates))
        preds = self.daily_prediction_.reindex(idx)
        if preds.isna().any():
            raise ValueError("Missing rolling predictions for some requested dates")
        preds.index = idx
        return preds

    def get_params(self) -> dict[str, int]:
        return {
            "window_size": int(self.window_size),
            "min_periods": int(self.min_periods),
        }

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GlobalRollingSeasonalPoissonModel:
    window_size: int = 7
    min_periods: int = 1
    seasonal_profile_: np.ndarray | None = None
    level_prediction_: pd.Series | None = None

    def fit(
        self,
        train_daily_mean_series: pd.Series,
        full_daily_mean_series: pd.Series,
    ) -> "GlobalRollingSeasonalPoissonModel":
        full_daily_mean_series = full_daily_mean_series.sort_index().astype(float)
        train_daily_mean_series = train_daily_mean_series.sort_index().astype(float)

        self.level_prediction_ = full_daily_mean_series.shift(1).rolling(
            window=int(self.window_size),
            min_periods=int(self.min_periods),
        ).mean()

        train_level = self.level_prediction_.reindex(train_daily_mean_series.index)
        if train_level.isna().any():
            raise ValueError("Missing rolling baseline predictions on train dates")

        train_dow = train_daily_mean_series.index.dayofweek.to_numpy(dtype=int)
        train_ratio = train_daily_mean_series.to_numpy(dtype=float) / np.clip(train_level.to_numpy(dtype=float), 1e-12, None)
        ratio_df = pd.DataFrame({"dow": train_dow, "ratio": train_ratio})
        ratio_by_dow = ratio_df.groupby("dow")["ratio"].mean().reindex(range(7), fill_value=1.0).to_numpy(dtype=float)

        exposure = ratio_df.groupby("dow").size().reindex(range(7), fill_value=0).to_numpy(dtype=float)
        weighted_mean = float(np.sum(ratio_by_dow * exposure) / max(exposure.sum(), 1.0))
        self.seasonal_profile_ = ratio_by_dow / max(weighted_mean, 1e-12)
        return self

    def predict_for_dates(self, dates) -> pd.Series:
        if self.level_prediction_ is None or self.seasonal_profile_ is None:
            raise ValueError("Model is not fitted")
        idx = pd.to_datetime(pd.Index(dates))
        level = self.level_prediction_.reindex(idx)
        if level.isna().any():
            raise ValueError("Missing rolling level predictions for some requested dates")
        dow = idx.dayofweek.to_numpy(dtype=int)
        pred = level.to_numpy(dtype=float) * self.seasonal_profile_[dow]
        return pd.Series(pred, index=idx)

    def get_params(self) -> dict[str, int | float | list[float]]:
        if self.seasonal_profile_ is None:
            raise ValueError("Model is not fitted")
        return {
            "window_size": int(self.window_size),
            "min_periods": int(self.min_periods),
            "seasonal_profile": [float(x) for x in self.seasonal_profile_],
        }

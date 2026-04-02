from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GlobalSeasonalPoissonModel:
    mu_: float | None = None
    seasonal_profile_: np.ndarray | None = None

    def fit(self, y_train, dow_train) -> "GlobalSeasonalPoissonModel":
        y_train = np.asarray(y_train, dtype=float)
        dow_train = np.asarray(dow_train, dtype=int)
        if y_train.size == 0:
            raise ValueError("Cannot fit seasonal Poisson model on empty data")
        if y_train.size != dow_train.size:
            raise ValueError("y_train and dow_train must have equal length")

        exposure = np.bincount(dow_train, minlength=7).astype(float)
        counts = np.bincount(dow_train, weights=y_train, minlength=7).astype(float)
        mean_by_dow = np.divide(counts, exposure, out=np.zeros(7, dtype=float), where=exposure > 0)

        self.mu_ = float(np.mean(y_train))
        profile = np.divide(mean_by_dow, max(self.mu_, 1e-12))
        profile = np.clip(profile, 1e-8, None)

        weighted_mean = float(np.sum(profile * exposure) / max(exposure.sum(), 1.0))
        self.seasonal_profile_ = profile / max(weighted_mean, 1e-12)
        return self

    def predict(self, dow_values) -> np.ndarray:
        if self.mu_ is None or self.seasonal_profile_ is None:
            raise ValueError("Model is not fitted")
        dow_values = np.asarray(dow_values, dtype=int)
        return self.mu_ * self.seasonal_profile_[dow_values]

    def get_params(self) -> dict[str, float | list[float]]:
        if self.mu_ is None or self.seasonal_profile_ is None:
            raise ValueError("Model is not fitted")
        return {
            "mu": float(self.mu_),
            "seasonal_profile": [float(x) for x in self.seasonal_profile_],
        }

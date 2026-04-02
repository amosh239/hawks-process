from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GlobalPoissonModel:
    mu_: float | None = None

    def fit(self, y_train) -> "GlobalPoissonModel":
        y_train = np.asarray(y_train, dtype=float)
        if y_train.size == 0:
            raise ValueError("Cannot fit Poisson model on empty data")
        self.mu_ = float(np.mean(y_train))
        return self

    def predict(self, n_obs: int) -> np.ndarray:
        if self.mu_ is None:
            raise ValueError("Model is not fitted")
        return np.full(int(n_obs), self.mu_, dtype=float)

    def get_params(self) -> dict[str, float]:
        if self.mu_ is None:
            raise ValueError("Model is not fitted")
        return {"mu": float(self.mu_)}

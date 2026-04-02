from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


@dataclass
class PersonalizedGammaPoissonScaler:
    alpha_: float | None = None
    beta_: float | None = None
    prior_mean_: float | None = None
    user_stats_: pd.DataFrame | None = None

    def fit(self, user_ids, y_train, base_lambda_train) -> "PersonalizedGammaPoissonScaler":
        frame = pd.DataFrame(
            {
                "user_id": np.asarray(user_ids),
                "y": np.asarray(y_train, dtype=float),
                "base_lambda": np.asarray(base_lambda_train, dtype=float),
            }
        )
        grouped = frame.groupby("user_id", as_index=True).agg(
            y_sum=("y", "sum"),
            exposure=("base_lambda", "sum"),
        )
        grouped = grouped[grouped["exposure"] > 0].copy()
        if grouped.empty:
            raise ValueError("No positive user exposures for Gamma-Poisson scaling")

        y = grouped["y_sum"].to_numpy(dtype=float)
        exposure = grouped["exposure"].to_numpy(dtype=float)

        def _neg_log_marginal(log_params: np.ndarray) -> float:
            alpha = float(np.exp(log_params[0]))
            beta = float(np.exp(log_params[1]))
            val = -np.sum(
                gammaln(y + alpha)
                - gammaln(alpha)
                - gammaln(y + 1.0)
                + alpha * np.log(beta)
                + y * np.log(np.clip(exposure, 1e-12, None))
                - (y + alpha) * np.log(beta + exposure)
            )
            return float(val)

        init = np.log([2.0, 2.0])
        try:
            res = minimize(_neg_log_marginal, init, method="L-BFGS-B")
            if not bool(res.success):
                alpha, beta = 1.0, 1.0
            else:
                alpha, beta = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
        except ValueError:
            alpha, beta = 1.0, 1.0

        grouped["mu_mle"] = grouped["y_sum"] / grouped["exposure"]
        grouped["mu_posterior_mean"] = (alpha + grouped["y_sum"]) / (beta + grouped["exposure"])

        self.alpha_ = alpha
        self.beta_ = beta
        self.prior_mean_ = float(alpha / beta)
        self.user_stats_ = grouped
        return self

    def predict_multiplier(self, user_ids, method: str = "posterior_mean") -> pd.Series:
        if self.user_stats_ is None or self.prior_mean_ is None:
            raise ValueError("Scaler is not fitted")

        user_ids = pd.Index(np.asarray(user_ids))
        if method == "posterior_mean":
            mapped = pd.Series(user_ids, index=range(len(user_ids))).map(self.user_stats_["mu_posterior_mean"])
            return mapped.fillna(self.prior_mean_)
        if method == "mle":
            mapped = pd.Series(user_ids, index=range(len(user_ids))).map(self.user_stats_["mu_mle"])
            return mapped.fillna(1.0)
        raise ValueError(f"Unknown method: {method}")

    def predict(self, user_ids, base_lambda, method: str = "posterior_mean") -> np.ndarray:
        multipliers = self.predict_multiplier(user_ids, method=method).to_numpy(dtype=float)
        return multipliers * np.asarray(base_lambda, dtype=float)

    def get_params(self) -> dict[str, float]:
        if self.alpha_ is None or self.beta_ is None or self.prior_mean_ is None:
            raise ValueError("Scaler is not fitted")
        return {
            "alpha": float(self.alpha_),
            "beta": float(self.beta_),
            "prior_mean": float(self.prior_mean_),
            "prior_variance": float(self.alpha_ / (self.beta_**2)),
        }

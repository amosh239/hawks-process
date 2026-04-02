from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import gammaln


def poisson_loglik(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-12, None)
    return float(np.sum(y_true * np.log(y_pred) - y_pred - gammaln(y_true + 1.0)))


def poisson_loglik_contrib(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-12, None)
    return y_true * np.log(y_pred) - y_pred - gammaln(y_true + 1.0)


def mean_poisson_nll(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    n_obs = max(int(y_true.size), 1)
    return float(-poisson_loglik(y_true, y_pred) / n_obs)


def mean_poisson_deviance(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-12, None)

    term = np.zeros_like(y_true, dtype=float)
    positive = y_true > 0
    term[positive] = y_true[positive] * np.log(y_true[positive] / y_pred[positive])
    deviance = 2.0 * np.sum(term - (y_true - y_pred))
    return float(deviance / max(int(y_true.size), 1))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def aggregate_bias(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred) - np.mean(y_true))


def relative_aggregate_bias(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    denom = max(float(np.mean(y_true)), 1e-12)
    return float(aggregate_bias(y_true, y_pred) / denom)


def evaluate_count_forecast(y_true, y_pred) -> dict[str, float]:
    return {
        "poisson_loglik": poisson_loglik(y_true, y_pred),
        "mean_poisson_nll": mean_poisson_nll(y_true, y_pred),
        "mean_poisson_deviance": mean_poisson_deviance(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "aggregate_bias": aggregate_bias(y_true, y_pred),
        "relative_aggregate_bias": relative_aggregate_bias(y_true, y_pred),
        "mean_target": float(np.mean(np.asarray(y_true, dtype=float))),
        "mean_prediction": float(np.mean(np.asarray(y_pred, dtype=float))),
    }


def aggregate_user_loglik(
    user_ids,
    y_true,
    y_pred,
    user_col: str = "user_id",
    ll_col: str = "poisson_loglik",
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            user_col: np.asarray(user_ids),
            ll_col: poisson_loglik_contrib(y_true, y_pred),
        }
    )
    return frame.groupby(user_col, as_index=False)[ll_col].sum()

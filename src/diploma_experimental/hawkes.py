from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


FEATURE_NAMES = [
    "searches",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "to_cart",
    "to_ord",
]

DEFAULT_HALF_LIVES = (1.0, 3.0, 7.0, 21.0)


@dataclass
class PooledAdditiveMultiKernelHawkesResult:
    alpha: np.ndarray
    beta: np.ndarray
    half_lives: np.ndarray
    feature_names: tuple[str, ...]
    success: bool

    def alpha_matrix(self) -> np.ndarray:
        return np.asarray(self.alpha, dtype=float).reshape(len(self.feature_names), len(self.half_lives))


def build_basis_states(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n_days, n_features = x.shape
    n_basis = int(beta.shape[0])
    states = np.zeros((n_days, n_features, n_basis), dtype=float)
    decay = np.exp(-beta).reshape(1, n_basis)
    for t in range(1, n_days):
        states[t] = states[t - 1] * decay + x[t - 1].reshape(n_features, 1)
    return states


def fit_pooled_additive_multi_kernel_hawkes(
    state_blocks: list[np.ndarray],
    y_blocks: list[np.ndarray],
    base_blocks: list[np.ndarray],
    half_lives: tuple[float, ...] = DEFAULT_HALF_LIVES,
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES),
    alpha_l2: float = 1e-4,
    max_iter: int = 120,
) -> PooledAdditiveMultiKernelHawkesResult:
    if not state_blocks:
        raise ValueError("No train blocks provided for Hawkes fit")

    x = np.vstack([np.asarray(block, dtype=np.float32) for block in state_blocks])
    y = np.concatenate([np.asarray(block, dtype=float) for block in y_blocks])
    base = np.concatenate([np.asarray(block, dtype=float) for block in base_blocks])

    n_params = int(x.shape[1])

    def _objective(alpha: np.ndarray) -> tuple[float, np.ndarray]:
        lam = np.clip(base + x @ alpha, 1e-8, None)
        nll = float(np.sum(lam - y * np.log(lam)) + float(alpha_l2) * np.sum(alpha**2))
        grad = x.T @ (1.0 - y / lam) + 2.0 * float(alpha_l2) * alpha
        return nll, grad

    init = np.full(n_params, 0.01, dtype=float)
    bounds = [(0.0, 10.0)] * n_params
    try:
        res = minimize(
            lambda a: _objective(a)[0],
            init,
            method="L-BFGS-B",
            jac=lambda a: _objective(a)[1],
            bounds=bounds,
            options={"maxiter": int(max_iter)},
        )
        alpha = np.asarray(res.x, dtype=float)
        success = bool(res.success)
    except ValueError:
        alpha = np.zeros(n_params, dtype=float)
        success = False

    beta = np.log(2.0) / np.asarray(half_lives, dtype=float)
    return PooledAdditiveMultiKernelHawkesResult(
        alpha=alpha,
        beta=beta,
        half_lives=np.asarray(half_lives, dtype=float),
        feature_names=tuple(feature_names),
        success=success,
    )


def predict_pooled_additive_multi_kernel_hawkes(
    model: PooledAdditiveMultiKernelHawkesResult,
    states: np.ndarray,
    base_lambda: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=float).reshape(len(states), -1)
    excitation = states @ np.asarray(model.alpha, dtype=float)
    lam = np.clip(np.asarray(base_lambda, dtype=float) + excitation, 1e-8, None)
    return lam, excitation

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


ALL_COUNT_FEATURE_NAMES = [
    "searches",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "to_cart",
    "to_ord",
]

# Main Hawkes experiments currently exclude search-attributed conversion counts,
# because they are almost fully duplicated by the remaining cart/order channels.
FEATURE_NAMES = [
    "searches",
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
    base_scale: float
    success: bool

    def alpha_matrix(self) -> np.ndarray:
        return np.asarray(self.alpha, dtype=float).reshape(len(self.feature_names), len(self.half_lives))


@dataclass
class JointHawkesResult:
    """Output of `fit_joint_hawkes` — joint Poisson MLE for `λ_t = lam_u · b_t + states · α`."""

    lam_u: np.ndarray
    alpha: np.ndarray
    train_loss: float
    converged: bool
    n_iter: int


@dataclass
class PooledHawkesResult:
    """Output of `fit_pooled_hawkes` — pooled fit `λ_t = c · b_t + states · α` (no per-user multiplier)."""

    c: float
    alpha: np.ndarray
    train_loss: float
    converged: bool
    n_iter: int


@dataclass
class UserScaleFitResult:
    scales: np.ndarray
    lower_bound_hits: int
    upper_bound_hits: int
    mean_scale: float
    median_scale: float


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
    learn_base_scale: bool = False,
    scale_l2: float = 0.0,
    scale_init: float = 1.0,
    alpha_init: np.ndarray | None = None,
    max_iter: int = 120,
) -> PooledAdditiveMultiKernelHawkesResult:
    if not state_blocks:
        raise ValueError("No train blocks provided for Hawkes fit")

    x = np.vstack([np.asarray(block, dtype=np.float32) for block in state_blocks])
    y = np.concatenate([np.asarray(block, dtype=float) for block in y_blocks])
    base = np.concatenate([np.asarray(block, dtype=float) for block in base_blocks])

    n_alpha = int(x.shape[1])

    def _objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        if learn_base_scale:
            scale = float(params[0])
            alpha = np.asarray(params[1:], dtype=float)
        else:
            scale = 1.0
            alpha = np.asarray(params, dtype=float)
        lam = np.clip(scale * base + x @ alpha, 1e-8, None)
        nll = float(
            np.sum(lam - y * np.log(lam))
            + float(alpha_l2) * np.sum(alpha**2)
            + float(scale_l2) * (scale - 1.0) ** 2
        )
        alpha_grad = x.T @ (1.0 - y / lam) + 2.0 * float(alpha_l2) * alpha
        if not learn_base_scale:
            return nll, alpha_grad
        scale_grad = float(np.sum(base * (1.0 - y / lam)) + 2.0 * float(scale_l2) * (scale - 1.0))
        grad = np.concatenate([[scale_grad], alpha_grad])
        return nll, grad

    if alpha_init is None:
        init_alpha = np.full(n_alpha, 0.01, dtype=float)
    else:
        init_alpha = np.asarray(alpha_init, dtype=float).reshape(-1)
        if init_alpha.size != n_alpha:
            raise ValueError("alpha_init has incompatible size")
    if learn_base_scale:
        init = np.concatenate([[float(scale_init)], init_alpha])
        bounds = [(0.1, 5.0)] + [(0.0, 10.0)] * n_alpha
    else:
        init = init_alpha
        bounds = [(0.0, 10.0)] * n_alpha
    try:
        res = minimize(
            lambda a: _objective(a)[0],
            init,
            method="L-BFGS-B",
            jac=lambda a: _objective(a)[1],
            bounds=bounds,
            options={"maxiter": int(max_iter)},
        )
        if learn_base_scale:
            base_scale = float(res.x[0])
            alpha = np.asarray(res.x[1:], dtype=float)
        else:
            base_scale = 1.0
            alpha = np.asarray(res.x, dtype=float)
        success = bool(res.success)
    except ValueError:
        base_scale = 1.0
        alpha = np.zeros(n_alpha, dtype=float)
        success = False

    beta = np.log(2.0) / np.asarray(half_lives, dtype=float)
    return PooledAdditiveMultiKernelHawkesResult(
        alpha=alpha,
        beta=beta,
        half_lives=np.asarray(half_lives, dtype=float),
        feature_names=tuple(feature_names),
        base_scale=base_scale,
        success=success,
    )


def fit_joint_hawkes(
    user_idx: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    states: np.ndarray,
    n_users: int,
    lambda_l2: float = 1.0,
    alpha_l2: float = 1e-4,
    lambda_init: np.ndarray | None = None,
    alpha_init: np.ndarray | None = None,
    max_iter: int = 300,
) -> JointHawkesResult:
    """Joint Poisson MLE fit of `λ_t = λ_{u(t)} · b_t + states_t · α`.

    Loss: Poisson NLL + `lambda_l2 · Σ(λ_u − 1)² + alpha_l2 · ‖α‖²`.
    L-BFGS-B with bounds `λ_u ∈ [0.001, 50]`, `α_j ∈ [0, 10]`.
    """
    user_idx = np.asarray(user_idx, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    n_alpha = int(states.shape[1])

    init_lam = np.ones(n_users) if lambda_init is None else np.asarray(lambda_init, dtype=float).copy()
    init_alpha = (
        np.full(n_alpha, 0.01, dtype=float)
        if alpha_init is None
        else np.asarray(alpha_init, dtype=float).copy()
    )
    init = np.concatenate([init_lam, init_alpha])
    bounds = [(0.001, 50.0)] * n_users + [(0.0, 10.0)] * n_alpha

    def fg(params: np.ndarray) -> tuple[float, np.ndarray]:
        lam_u = params[:n_users]
        alpha = params[n_users:]
        mu = np.clip(lam_u[user_idx] * b + states @ alpha, 1e-8, None)
        nll = float(
            np.sum(mu - y * np.log(mu))
            + lambda_l2 * np.sum((lam_u - 1.0) ** 2)
            + alpha_l2 * np.sum(alpha ** 2)
        )
        residual = 1.0 - y / mu
        grad_lam = np.zeros(n_users, dtype=np.float64)
        np.add.at(grad_lam, user_idx, residual * b)
        grad_lam += 2.0 * lambda_l2 * (lam_u - 1.0)
        grad_alpha = states.T @ residual + 2.0 * alpha_l2 * alpha
        return nll, np.concatenate([grad_lam, grad_alpha])

    res = minimize(
        lambda p: fg(p)[0], init, method="L-BFGS-B",
        jac=lambda p: fg(p)[1], bounds=bounds, options={"maxiter": max_iter},
    )
    return JointHawkesResult(
        lam_u=np.asarray(res.x[:n_users], dtype=float),
        alpha=np.asarray(res.x[n_users:], dtype=float),
        train_loss=float(res.fun),
        converged=bool(res.success),
        n_iter=int(getattr(res, "nit", 0)),
    )


def fit_pooled_hawkes(
    y: np.ndarray,
    b: np.ndarray,
    states: np.ndarray,
    alpha_l2: float = 1e-4,
    scale_l2: float = 10.0,
    c_init: float = 1.0,
    alpha_init_value: float = 0.01,
    max_iter: int = 300,
) -> PooledHawkesResult:
    """Pooled fit of `λ_t = c · b_t + states_t · α` (single global scale, no per-user multiplier).

    Loss: Poisson NLL + `alpha_l2 · ‖α‖² + scale_l2 · (c − 1)²`.
    L-BFGS-B with bounds `c ∈ [0.001, 50]`, `α_j ∈ [0, 10]`.
    """
    y = np.asarray(y, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    n_alpha = int(states.shape[1])

    init = np.concatenate([[float(c_init)], np.full(n_alpha, float(alpha_init_value))])
    bounds = [(0.001, 50.0)] + [(0.0, 10.0)] * n_alpha

    def fg(params: np.ndarray) -> tuple[float, np.ndarray]:
        c = float(params[0])
        alpha = params[1:]
        lam = np.clip(c * b + states @ alpha, 1e-8, None)
        nll = float(
            np.sum(lam - y * np.log(lam))
            + alpha_l2 * np.sum(alpha ** 2)
            + scale_l2 * (c - 1.0) ** 2
        )
        residual = 1.0 - y / lam
        a_grad = states.T @ residual + 2.0 * alpha_l2 * alpha
        c_grad = float(np.sum(b * residual) + 2.0 * scale_l2 * (c - 1.0))
        return nll, np.concatenate([[c_grad], a_grad])

    res = minimize(
        lambda p: fg(p)[0], init, method="L-BFGS-B",
        jac=lambda p: fg(p)[1], bounds=bounds, options={"maxiter": max_iter},
    )
    return PooledHawkesResult(
        c=float(res.x[0]),
        alpha=np.asarray(res.x[1:], dtype=float),
        train_loss=float(res.fun),
        converged=bool(res.success),
        n_iter=int(getattr(res, "nit", 0)),
    )


def predict_pooled_additive_multi_kernel_hawkes(
    model: PooledAdditiveMultiKernelHawkesResult,
    states: np.ndarray,
    base_lambda: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=float).reshape(len(states), -1)
    excitation = states @ np.asarray(model.alpha, dtype=float)
    lam = np.clip(float(model.base_scale) * np.asarray(base_lambda, dtype=float) + excitation, 1e-8, None)
    return lam, excitation


def solve_user_scale_mle(
    y: np.ndarray,
    base_lambda: np.ndarray,
    excitation: np.ndarray,
    lower: float = 0.0,
    upper: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 80,
) -> float:
    y = np.asarray(y, dtype=float)
    base = np.asarray(base_lambda, dtype=float)
    excitation = np.asarray(excitation, dtype=float)

    if y.size == 0:
        return 1.0

    def grad(scale: float) -> float:
        lam = np.clip(scale * base + excitation, 1e-8, None)
        return float(np.sum(base * (1.0 - y / lam)))

    g_lo = grad(lower)
    if g_lo >= 0.0:
        return float(lower)
    g_hi = grad(upper)
    if g_hi <= 0.0:
        return float(upper)

    lo = float(lower)
    hi = float(upper)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        g_mid = grad(mid)
        if abs(g_mid) <= tol or (hi - lo) <= tol:
            return float(mid)
        if g_mid > 0.0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def fit_user_scales_with_fixed_excitation(
    y_blocks: list[np.ndarray],
    base_blocks: list[np.ndarray],
    excitation_blocks: list[np.ndarray],
    init_scale: float = 1.0,
    lower: float = 0.0,
    upper: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 80,
) -> UserScaleFitResult:
    scales = []
    lower_hits = 0
    upper_hits = 0
    for y, base, excitation in zip(y_blocks, base_blocks, excitation_blocks):
        if np.asarray(y).size == 0:
            scale = float(init_scale)
            scales.append(scale)
            continue
        scale = solve_user_scale_mle(
            y=y,
            base_lambda=base,
            excitation=excitation,
            lower=lower,
            upper=upper,
            tol=tol,
            max_iter=max_iter,
        )
        if scale <= lower + max(tol, 1e-10):
            lower_hits += 1
        if scale >= upper - max(tol, 1e-10):
            upper_hits += 1
        scales.append(scale)

    if not scales:
        return UserScaleFitResult(
            scales=np.empty(0, dtype=float),
            lower_bound_hits=0,
            upper_bound_hits=0,
            mean_scale=float(init_scale),
            median_scale=float(init_scale),
        )

    scales_arr = np.asarray(scales, dtype=float)
    return UserScaleFitResult(
        scales=scales_arr,
        lower_bound_hits=int(lower_hits),
        upper_bound_hits=int(upper_hits),
        mean_scale=float(scales_arr.mean()),
        median_scale=float(np.median(scales_arr)),
    )

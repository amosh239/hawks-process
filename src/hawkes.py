import numpy as np
from scipy.optimize import minimize
from .utils import integrate_intensity


class HawkesExp:
    """Univariate Hawkes with exponential kernel and optional baseline lambda0(t)."""

    def __init__(self, baseline=None, penalty=0.0):
        self.baseline = baseline
        self.penalty = penalty
        self.params = None  # [mu_scale or mu_const, alpha, beta]

    def _base_intensity(self, t, mu):
        if self.baseline is None:
            return mu
        return mu * self.baseline.get_intensity(t)

    def _base_integral(self, mu, t_start, t_end):
        if self.baseline is None:
            return mu * (t_end - t_start).total_seconds() / 3600.0
        return mu * integrate_intensity(self.baseline, t_start, t_end)

    def _nll(self, params, times, t_start, t_end):
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= 0:
            return 1e9

        t0 = t_start
        rel = np.array([(t - t0).total_seconds() / 3600.0 for t in times])
        log_sum = np.log(max(self._base_intensity(times[0], mu), 1e-12))
        R = 0.0

        for i in range(1, len(times)):
            dt = rel[i] - rel[i - 1]
            R = np.exp(-beta * dt) * (R + alpha)
            lam = self._base_intensity(times[i], mu) + R
            if lam <= 0:
                return 1e9
            log_sum += np.log(lam)

        T_max = (t_end - t0).total_seconds() / 3600.0
        kernel_int = (alpha / beta) * np.sum(1 - np.exp(-beta * (T_max - rel)))
        base_int = self._base_integral(mu, t0, t_end)

        return -(log_sum - base_int - kernel_int) + self.penalty * (alpha ** 2)

    def fit(self, times):
        if len(times) == 0:
            self.params = [1.0, 0.0, 1.0]
            return self
        t_end = times[-1]
        duration = max(1e-6, (t_end - times[0]).total_seconds() / 3600.0)
        avg_rate = len(times) / duration

        init = [max(1e-3, avg_rate), 0.1, 1.0]
        bounds = [(1e-6, avg_rate * 10 + 10), (0.0, max(10.0, avg_rate * 50)), (1e-4, 20.0)]

        res = minimize(lambda p: self._nll(p, times, times[0], t_end), init, method="L-BFGS-B", bounds=bounds)
        self.params = res.x.tolist()
        return self

    def nll(self, times, types=None, t_end=None, window_start=None):
        if self.params is None:
            raise RuntimeError("Model not fitted")
        t_end = t_end or times[-1]
        t_start = window_start or times[0]
        return self._nll(self.params, times, t_start, t_end)

    def get_intensity(self, t, history, history_types=None):
        mu, alpha, beta = self.params
        lam = self._base_intensity(t, mu)
        if len(history) == 0:
            return lam
        deltas = np.array([(t - h).total_seconds() / 3600.0 for h in history])
        mask = deltas > 0
        if np.any(mask):
            lam += alpha * np.sum(np.exp(-beta * deltas[mask]))
        return lam


class PurchaseTargetHawkes:
    """Purchase intensity depends on multi-type history (exp kernels per type)."""

    def __init__(self, baseline, n_types=3, target_type=2):
        self.baseline = baseline
        self.n_types = n_types
        self.target_type = target_type
        self.params = None  # [mu] + alphas + betas

    def _base_intensity(self, t, mu):
        return mu * self.baseline.get_intensity(t)

    def _base_integral(self, mu, t_start, t_end):
        return mu * integrate_intensity(self.baseline, t_start, t_end)

    def _nll(self, params, times, types, t_start, t_end):
        mu = params[0]
        alphas = np.array(params[1 : 1 + self.n_types])
        betas = np.array(params[1 + self.n_types :])

        if mu <= 0 or np.any(alphas < 0) or np.any(betas <= 0):
            return 1e9

        t0 = t_start
        rel = np.array([(t - t0).total_seconds() / 3600.0 for t in times])
        arr_types = np.array(types)
        r = np.zeros(self.n_types)
        log_sum = 0.0

        for i, (tau, ev_type) in enumerate(zip(rel, types)):
            if i > 0:
                dt = tau - rel[i - 1]
                r *= np.exp(-betas * dt)
            lam = self._base_intensity(times[i], mu) + r.sum()
            if ev_type == self.target_type:
                if lam <= 0:
                    return 1e9
                # Numeric floor: keeps logs finite without changing the model for reasonable lambdas.
                log_sum += np.log(max(lam, 1e-12))
            r[ev_type] += alphas[ev_type]

        T_max = (t_end - t0).total_seconds() / 3600.0
        kernel_int = 0.0
        for ev_type in range(self.n_types):
            t_e = rel[arr_types == ev_type]
            if len(t_e) == 0:
                continue
            kernel_int += (alphas[ev_type] / betas[ev_type]) * np.sum(1 - np.exp(-betas[ev_type] * (T_max - t_e)))

        base_int = self._base_integral(mu, t0, t_end)
        return -(log_sum - base_int - kernel_int)

    def fit(self, times, types):
        if len(times) == 0:
            self.params = [1.0] + [0.0] * (2 * self.n_types)
            return self
        t_end = times[-1]
        t_start = times[0]
        duration = max(1e-6, (t_end - times[0]).total_seconds() / 3600.0)
        purchases = sum(1 for t in types if t == self.target_type)
        avg_rate = purchases / duration if duration > 0 else 0.1

        init_mu = max(1e-3, avg_rate)
        init = [init_mu] + [0.01] * self.n_types + [1.0] * self.n_types
        bounds = [(1e-6, init_mu * 20 + 5)]
        bounds += [(0.0, 5.0)] * self.n_types
        bounds += [(1e-4, 20.0)] * self.n_types

        res = minimize(lambda p: self._nll(p, times, types, t_start, t_end), init, method="L-BFGS-B", bounds=bounds)
        self.params = res.x.tolist()
        return self

    def nll(self, times, types=None, t_end=None, window_start=None):
        if self.params is None:
            raise RuntimeError("Model not fitted")
        if types is None:
            raise ValueError("Types are required for purchase-target model")
        t_end = t_end or times[-1]
        t_start = window_start or times[0]
        return self._nll(self.params, times, types, t_start, t_end)

    def get_intensity(self, t, history, history_types):
        mu = self.params[0]
        alphas = np.array(self.params[1 : 1 + self.n_types])
        betas = np.array(self.params[1 + self.n_types :])
        lam = self._base_intensity(t, mu)
        for h, h_type in zip(history, history_types):
            dt = (t - h).total_seconds() / 3600.0
            if dt <= 0:
                continue
            lam += alphas[h_type] * np.exp(-betas[h_type] * dt)
        return lam

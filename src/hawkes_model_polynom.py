import numpy as np
from scipy.optimize import minimize

class HawkesExp:
    def __init__(self, baseline_model=None, penalty_weight=0.0, kernel='exp'):
        self.baseline = baseline_model
        self.penalty_weight = penalty_weight
        self.kernel = kernel
        self.params = None 

    def _unpack(self, p):
        if self.kernel == 'power':
            return p[0], p[1], p[2], p[3]
        return p[0], p[1], p[2], 1.0 

    def _get_base_mu(self, t, mu_param):
        if self.baseline:
            return self.baseline.get_intensity(t) * mu_param
        return mu_param

    def _nll_calc(self, params, times, T_max):
        mu_param, alpha, beta, delta = self._unpack(params)

        if alpha < 0 or beta <= (1.001 if self.kernel == 'power' else 0): return 1e10
        if self.baseline and mu_param < 0: return 1e10
        if not self.baseline and mu_param <= 0: return 1e10
        if self.kernel == 'power' and delta <= 0: return 1e10

        t_arr = np.array([(t - times[0]).total_seconds()/3600 for t in times])
        
        log_term = 0
        
        base_0 = self._get_base_mu(times[0], mu_param)
        log_term += np.log(max(base_0, 1e-9))

        if self.kernel == 'exp':
            dt = np.diff(t_arr)
            R = 0
            for i in range(1, len(t_arr)):
                R = np.exp(-beta * dt[i-1]) * (R + alpha)
                lam = self._get_base_mu(times[i], mu_param) + R
                log_term += np.log(max(lam, 1e-9))
            integral_kernel = (alpha / beta) * np.sum(1 - np.exp(-beta * (T_max - t_arr)))

        elif self.kernel == 'power':
            dt_mat = t_arr[:, None] - t_arr[None, :]
            mask = dt_mat > 0
            
            R_vec = np.zeros(len(t_arr))
            k_mat = np.zeros_like(dt_mat)
            k_mat[mask] = alpha / np.power(delta + dt_mat[mask], beta)
            R_vec = k_mat.sum(axis=1)

            mus = np.array([self._get_base_mu(t, mu_param) for t in times])
            lams = mus + R_vec
            log_term += np.sum(np.log(np.maximum(lams[1:], 1e-9)))

            term1 = delta**(1.0 - beta)
            term2 = (delta + T_max - t_arr)**(1.0 - beta)
            integral_kernel = (alpha / (beta - 1.0)) * np.sum(term1 - term2)

        if self.baseline:
            rate = self.baseline.user_rate if hasattr(self.baseline, 'user_rate') else np.mean(self.baseline.rates)
            integral_base = rate * T_max * mu_param
        else:
            integral_base = mu_param * T_max

        return -(log_term - integral_base - integral_kernel) + self.penalty_weight * (alpha**2)

    def fit(self, times):
        T_max = max((times[-1] - times[0]).total_seconds() / 3600, 1e-3)
        avg = len(times) / T_max
        a_max = max(10.0, avg * 100.0)

        
        init = [1.0, 1e-5, 1.0] if self.baseline else [avg, 1e-5, 1.0]
        b_mu = (0.01, 5.0) if self.baseline else (1e-6, avg * 5.0)
        bounds = [b_mu, (1e-6, a_max), (0.01, 10.0)]

        if self.kernel == 'power':
            init.append(1.0)
            bounds[2] = (1.01, 10.0)
            bounds.append((0.1, 24.0))

        res = minimize(lambda p: self._nll_calc(p, times, T_max), 
                       init, method='L-BFGS-B', bounds=bounds)
        self.params = res.x
        return self

    def nll(self, times, T_max):
        old_p = self.penalty_weight
        self.penalty_weight = 0
        val = self._nll_calc(self.params, times, T_max)
        self.penalty_weight = old_p
        return val

    def get_intensity(self, t, history):
        mu_p, alpha, beta, delta = self._unpack(self.params)
        mu = self._get_base_mu(t, mu_p)
        
        dt = (t.timestamp() - np.array([h.timestamp() for h in history])) / 3600.0
        valid = dt[dt > 0]
        
        R = 0
        if len(valid) > 0:
            if self.kernel == 'exp':   R = np.sum(alpha * np.exp(-beta * valid))
            elif self.kernel == 'power': R = np.sum(alpha / np.power(delta + valid, beta))
        return mu + R
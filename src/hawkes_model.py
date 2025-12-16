import numpy as np
from scipy.optimize import minimize

class HawkesExp:
    def __init__(self, baseline_model=None, penalty_weight=0.0):
        self.baseline = baseline_model
        self.penalty_weight = penalty_weight
        self.params = None 

    def _nll_calc(self, params, times, T_max):
        if self.baseline:
            mu_scale, alpha, beta = params
        else:
            mu_const, alpha, beta = params

        if alpha < 0 or beta <= 0: return 1e10
        if self.baseline and mu_scale < 0: return 1e10
        if not self.baseline and mu_const <= 0: return 1e10
        
        t_arr = np.array([(t - times[0]).total_seconds()/3600 for t in times])
        dt = np.diff(t_arr)
        
        R = 0
        log_term = 0
        
        if self.baseline:
            base_int = self.baseline.get_intensity(times[0]) * mu_scale
            log_term += np.log(base_int if base_int > 1e-9 else 1e-9)
        else:
            log_term += np.log(mu_const)

        for i in range(1, len(t_arr)):
            R = np.exp(-beta * dt[i-1]) * (R + alpha)
            
            if self.baseline:
                mu_val = self.baseline.get_intensity(times[i]) * mu_scale
            else:
                mu_val = mu_const
            
            lam = mu_val + R
            if lam <= 1e-9: return 1e10
            log_term += np.log(lam)
            
        integral_hawkes = (alpha / beta) * np.sum(1 - np.exp(-beta * (T_max - t_arr)))
        
        if self.baseline:
            if hasattr(self.baseline, 'user_rate'):
                integral_base = self.baseline.user_rate * T_max * mu_scale
            else:
                integral_base = np.mean(self.baseline.rates) * T_max * mu_scale
        else:
            integral_base = mu_const * T_max
            
        nll = -(log_term - integral_base - integral_hawkes)

        reg_term = self.penalty_weight * (alpha ** 2)
        
        return nll + reg_term

    def fit(self, times):
        T_max = (times[-1] - times[0]).total_seconds() / 3600
        if T_max < 1e-3: T_max = 1e-3
        
        avg_rate = len(times) / T_max
        alpha_max = max(10.0, avg_rate * 100.0)
        
        if self.baseline:
            init = [1.0, 1e-5, 1.0]
            bounds = ((0.01, 5.0), (1e-6, alpha_max), (0.01, 10.0))
        else:
            init = [avg_rate, 1e-5, 1.0]
            bounds = ((1e-6, avg_rate * 5.0), (1e-6, alpha_max), (0.01, 10.0))
            
        res = minimize(lambda p: self._nll_calc(p, times, T_max), 
                       init, method='L-BFGS-B', bounds=bounds)
        self.params = res.x
        return self
    
    def nll(self, times, T_max_hours):
        saved = self.penalty_weight
        self.penalty_weight = 0
        val = self._nll_calc(self.params, times, T_max_hours)
        self.penalty_weight = saved
        return val

    def get_intensity(self, t, history):
        if self.baseline:
            mu_scale, alpha, beta = self.params
            mu = self.baseline.get_intensity(t) * mu_scale 
        else:
            mu, alpha, beta = self.params
        
        t_sec = t.timestamp()
        hist_sec = np.array([h.timestamp() for h in history])
        dt = t_sec - hist_sec
        mask = dt > 0
        
        R = 0
        if np.any(mask):
            R = np.sum(alpha * np.exp(-beta * (dt[mask] / 3600.0)))
            
        return mu + R
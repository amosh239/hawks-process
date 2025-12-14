import numpy as np
from scipy.optimize import minimize

class HawkesExp:
    def __init__(self, baseline_model=None, penalty_weight=10.0):
        """
        penalty_weight: Штраф за величину alpha. 
        Чем он больше, тем сильнее модель стремится стать обычным Пуассоном (alpha -> 0),
        если в данных нет явных кластеров.
        """
        self.baseline = baseline_model
        self.penalty_weight = penalty_weight
        self.params = None 

    def _nll_calc(self, params, times, T_max):
        if self.baseline:
            mu_const = 0
            alpha, beta = params
        else:
            mu_const, alpha, beta = params

        # Жесткая защита
        if alpha < 0 or beta <= 0: return 1e10
        if not self.baseline and mu_const <= 0: return 1e10
        
        t_arr = np.array([(t - times[0]).total_seconds()/3600 for t in times])
        dt = np.diff(t_arr)
        
        # --- LOG-LIKELIHOOD CALCULATION ---
        R = 0
        log_term = 0
        
        # Log(lambda) для первого события
        if self.baseline:
            base_int = self.baseline.get_intensity(times[0])
            log_term += np.log(base_int if base_int > 1e-9 else 1e-9)
        else:
            log_term += np.log(mu_const)

        for i in range(1, len(t_arr)):
            R = np.exp(-beta * dt[i-1]) * (R + alpha)
            mu_val = self.baseline.get_intensity(times[i]) if self.baseline else mu_const
            
            lam = mu_val + R
            if lam <= 1e-9: return 1e10
            log_term += np.log(lam)
            
        # Integrals
        integral_hawkes = (alpha / beta) * np.sum(1 - np.exp(-beta * (T_max - t_arr)))
        
        if self.baseline:
            if hasattr(self.baseline, 'user_rate'):
                integral_base = self.baseline.user_rate * T_max
            else:
                integral_base = np.mean(self.baseline.rates) * T_max
        else:
            integral_base = mu_const * T_max
            
        nll = -(log_term - integral_base - integral_hawkes)
        reg_term = self.penalty_weight * alpha
        
        return nll + reg_term

    def fit(self, times):
        T_max = (times[-1] - times[0]).total_seconds() / 3600
        if T_max < 1e-3: T_max = 1e-3
        
        avg_rate = len(times) / T_max
        
        alpha_max = max(4, avg_rate * 100.0)

        beta_min = 0.1 
        beta_max = 5.0
        
        if self.baseline:
            # Стартуем с alpha=0 (будь Пуассоном по умолчанию!)
            init = [1e-5, 1.0] 
            bounds = ((1e-6, alpha_max), (beta_min, beta_max))
        else:
            init = [avg_rate, 1e-5, 1.0]
            bounds = ((1e-6, avg_rate * 5.0), (1e-6, alpha_max), (beta_min, beta_max))
            
        res = minimize(lambda p: self._nll_calc(p, times, T_max), 
                       init, method='L-BFGS-B', bounds=bounds)
        
        self.params = res.x
        return self
    
    def nll(self, times, T_max_hours):
        saved_penalty = self.penalty_weight
        self.penalty_weight = 0
        val = self._nll_calc(self.params, times, T_max_hours)
        self.penalty_weight = saved_penalty
        return val

    def get_intensity(self, t, history):
        if self.baseline:
            mu = self.baseline.get_intensity(t)
            alpha, beta = self.params
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
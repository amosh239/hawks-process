import numpy as np
from scipy.optimize import minimize

class GlobalSeasonalProfile:
    def __init__(self, period=24):
        self.period = period
        self.profile = None 

    def fit(self, all_times_list):
        if len(all_times_list) > 0 and isinstance(all_times_list[0], list):
             all_times = [t for seq in all_times_list for t in seq]
        else:
             all_times = all_times_list
             
        hours = [t.hour for t in all_times]
        counts = np.bincount(hours, minlength=self.period)
        
        if counts.sum() > 0:
            self.profile = counts / counts.sum()
        else:
            self.profile = np.ones(self.period) / self.period
            
        self.profile = (counts + 1) / (counts.sum() + self.period)
        return self

    def get_rate(self, t, user_average_rate):
        return user_average_rate * self.profile[t.hour] * self.period

class SeasonalPoisson:
    def __init__(self, period=24):
        self.period = period
        self.rates = None

    def fit(self, times):
        hours = [t.hour for t in times]
        counts = np.bincount(hours, minlength=self.period)
        total_days = max(1, (times[-1] - times[0]).total_seconds() / (3600 * 24))
        self.rates = counts / total_days 
        self.rates[self.rates == 0] = 1e-5
        return self

    def get_intensity(self, t, history=None):
        return self.rates[t.hour]
    
    def nll(self, times, T_max_hours):
        log_sum = sum(np.log(self.rates[t.hour]) for t in times)
        integral = np.mean(self.rates) * T_max_hours
        return -(log_sum - integral)


class HawkesExp:
    def __init__(self, baseline_model=None):
        self.baseline = baseline_model
        self.params = None 

    def _nll_calc(self, params, times, T_max):
        if self.baseline:
            mu_const = 0
            alpha, beta = params
        else:
            mu_const, alpha, beta = params

        if alpha < 0 or beta <= 0: return 1e10
        if not self.baseline and mu_const <= 0: return 1e10
        
        t_arr = np.array([(t - times[0]).total_seconds()/3600 for t in times])
        dt = np.diff(t_arr)
        
        R = 0
        log_term = 0
        
        if self.baseline:
            log_term += np.log(self.baseline.get_intensity(times[0]))
        else:
            log_term += np.log(mu_const)

        for i in range(1, len(t_arr)):
            R = np.exp(-beta * dt[i-1]) * (R + alpha)
            mu_val = self.baseline.get_intensity(times[i]) if self.baseline else mu_const
            lam = mu_val + R
            if lam <= 0: return 1e10
            log_term += np.log(lam)
            
        integral_hawkes = (alpha / beta) * np.sum(1 - np.exp(-beta * (T_max - t_arr)))
        
        if self.baseline:
            integral_base = np.mean(self.baseline.rates) * T_max 
        else:
            integral_base = mu_const * T_max
            
        return -(log_term - integral_base - integral_hawkes)

    def fit(self, times):
        T_max = (times[-1] - times[0]).total_seconds() / 3600
        if self.baseline:
            init = [0.5, 1.0]
            bounds = ((1e-5, None), (1e-5, None))
        else:
            init = [0.1, 0.5, 1.0]
            bounds = ((1e-5, None), (1e-5, None), (1e-5, None))
            
        res = minimize(lambda p: self._nll_calc(p, times, T_max), 
                       init, method='L-BFGS-B', bounds=bounds)
        self.params = res.x
        return self
    
    def nll(self, times, T_max_hours):
        return self._nll_calc(self.params, times, T_max_hours)

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
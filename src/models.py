import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
        self.profile = (counts + 1) / (counts.sum() + self.period)
        return self
    
    def get_val(self, hour):
        return self.profile[hour]

class GlobalSeasonalPoisson:
    def __init__(self, global_profile, user_train_times):
        self.glob = global_profile
        duration = (user_train_times[-1] - user_train_times[0]).total_seconds() / 3600
        if duration < 1: duration = 1
        self.user_rate = len(user_train_times) / duration
        
    def get_intensity(self, t, history=None):
        return self.user_rate * self.glob.get_val(t.hour) * 24
    
    def nll(self, times, T_max_hours):
        log_sum = 0
        for t in times:
            lam = self.get_intensity(t)
            log_sum += np.log(lam)
        integral = self.user_rate * T_max_hours
        return -(log_sum - integral)

class ScaledPoisson:
    def __init__(self, baseline):
        self.baseline = baseline
        self.mu = 1.0
        self.kernel = 'scaled' 
        self.penalty_weight = 0

    def fit(self, times):
        ts = times.values if hasattr(times, 'values') else times
        
        if len(ts) < 2:
            self.mu = 1.0
            return self

        t_start, t_end = ts[0], ts[-1]
        T_hours = (t_end - t_start).total_seconds() / 3600
        if T_hours < 1e-3: T_hours = 1e-3
        
        check_points = pd.date_range(start=t_start, end=t_end, periods=10)
        base_rates = [self.baseline.get_intensity(t) for t in check_points]
        base_avg = np.mean(base_rates)
        
        integral_base = base_avg * T_hours
        
        if integral_base > 1e-9:
            self.mu = len(ts) / integral_base
        else:
            self.mu = 1.0
            
        return self

    def get_intensity(self, t, history=None):
        return self.baseline.get_intensity(t) * self.mu
    


    def nll(self, times, T_max=None):
        ts = times.values if hasattr(times, 'values') else times
        if len(ts) == 0: return 0.0
        
        t_start, t_end = ts[0], ts[-1]
        T_hours = (t_end - t_start).total_seconds() / 3600
        if T_hours < 1e-3: T_hours = 1e-3
        
        check_points = pd.date_range(t_start, t_end, periods=10)
        base_rates = [self.baseline.get_intensity(t) for t in check_points]
        integral = np.mean(base_rates) * T_hours * self.mu
        
        log_sum = 0
        for t in ts:
            lam = self.get_intensity(t)
            log_sum += np.log(lam) if lam > 1e-9 else -10
            
        return -(log_sum - integral)

    @property
    def params(self):
        return [self.mu, 0.0, 1.0, 1.0]

    def get_params_dict(self):
        return {
            'mu_scale': self.mu, 
            'alpha': 0.0, 
            'beta': 1.0, 
            'delta': 1.0
        }

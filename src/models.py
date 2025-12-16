import numpy as np
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

# --- Global Models ---
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

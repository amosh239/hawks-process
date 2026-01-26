import numpy as np
import pandas as pd
from .utils import integrate_intensity


class HourlySeasonalPoisson:
    """Piecewise-constant Poisson by hour-of-day."""

    def __init__(self):
        self.rates = np.ones(24, dtype=float)

    def fit(self, times):
        if len(times) == 0:
            return self
        hours = [t.hour for t in times]
        counts = np.bincount(hours, minlength=24)
        # Avoid explosive rates when we only have 0-1 events (or a very short span).
        span_days = max(1.0, (times[-1] - times[0]).total_seconds() / 86400.0)
        self.rates = counts / span_days
        self.rates[self.rates < 1e-8] = 1e-8
        return self

    def get_intensity(self, t, history=None):
        return float(self.rates[t.hour])


class ScaledBaseline:
    """lambda(t) = mu * lambda0(t); mu via exact MLE."""

    def __init__(self, baseline):
        self.base = baseline
        self.mu = 1.0

    def fit(self, times, window_start=None, window_end=None):
        if len(times) == 0:
            self.mu = 1.0
            return self

        start = window_start or times[0]
        end = window_end or times[-1]
        integral = integrate_intensity(self.base, start, end)
        self.mu = len(times) / integral if integral > 0 else 1.0
        return self

    def get_intensity(self, t, history=None):
        return self.mu * self.base.get_intensity(t)

    def nll(self, times, types=None, t_end=None, window_start=None):
        if len(times) == 0:
            return 0.0
        t_end = t_end or times[-1]
        t_start = window_start or times[0]
        log_sum = sum(np.log(max(self.get_intensity(t), 1e-12)) for t in times)
        integral = self.mu * integrate_intensity(self.base, t_start, t_end)
        return -(log_sum - integral)

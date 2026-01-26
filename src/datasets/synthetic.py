import numpy as np
import pandas as pd


class HawkesSimulator:
    """Simple synthetic Hawkes generator with sinusoidal baseline."""

    def __init__(self, t_max_hours=24 * 7):
        self.t_max_hours = t_max_hours

    def _baseline(self, t_hours, mu_level):
        cycle = np.sin(2 * np.pi * (t_hours % 24) / 24 - np.pi / 2) + 1.0
        return mu_level * (0.5 + 0.25 * cycle)

    def simulate(self, alpha, beta, mu_level, rng):
        t = 0.0
        r = 0.0
        events = []

        while t < self.t_max_hours:
            lam_bar = self._baseline(t, mu_level) * 1.5 + r
            step = rng.exponential(1.0 / max(lam_bar, 1e-6))
            t += step
            if t >= self.t_max_hours:
                break

            r *= np.exp(-beta * step)
            lam_true = self._baseline(t, mu_level) + r
            if rng.random() < lam_true / max(lam_bar, 1e-6):
                events.append(t)
                r += alpha

        start = pd.Timestamp("2024-01-01")
        return [start + pd.Timedelta(hours=h) for h in events]


def generate_synthetic_sequences(n_users=50, seed=42, t_max_hours=24 * 7):
    rng = np.random.default_rng(seed)
    sim = HawkesSimulator(t_max_hours=t_max_hours)
    sequences, truth = [], []

    for idx in range(n_users):
        beta = rng.uniform(1.0, 5.0)
        ratio = rng.uniform(0.0, 0.8)
        alpha = beta * ratio
        mu = rng.uniform(0.05, 0.3)

        times = sim.simulate(alpha=alpha, beta=beta, mu_level=mu, rng=rng)
        if len(times) < 5:
            continue

        sequences.append({"id": f"user_{idx}", "times": times})
        truth.append(
            {
                "user_id": f"user_{idx}",
                "alpha": alpha,
                "beta": beta,
                "mu": mu,
                "branching_ratio": ratio,
                "events": len(times),
            }
        )

    return sequences, pd.DataFrame(truth)

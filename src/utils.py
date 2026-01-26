import pandas as pd
from pathlib import Path


def temporal_train_test_split(times, types=None, train_ratio=0.8):
    """Time-ordered split; keeps alignment between times and optional types."""
    if len(times) == 0:
        return [], [], None, None

    t_start = times[0]
    duration_sec = (times[-1] - t_start).total_seconds()
    split_point = t_start + pd.Timedelta(seconds=duration_sec * train_ratio)

    train_times, test_times = [], []
    train_types, test_types = [], []

    for idx, t in enumerate(times):
        target = train_times if t <= split_point else test_times
        target.append(t)
        if types is not None:
            (train_types if t <= split_point else test_types).append(types[idx])

    return train_times, test_times, train_types or None, test_types or None


def integrate_intensity(baseline, t_start, t_end, step_minutes=5):
    """Integrate lambda0(t) dt over [t_start, t_end]. Exact for hourly piecewise."""
    if t_start >= t_end:
        return 0.0

    # Exact path if baseline exposes hourly rates
    rates = getattr(baseline, "rates", None)
    if rates is not None and len(rates) == 24:
        total = 0.0
        current = t_start
        while current < t_end:
            hour_end = (current.floor("h") + pd.Timedelta(hours=1))
            seg_end = min(hour_end, t_end)
            hours = (seg_end - current).total_seconds() / 3600.0
            total += float(rates[current.hour]) * hours
            current = seg_end
        return total

    # Fallback numeric grid
    step = pd.Timedelta(minutes=step_minutes)
    current = t_start
    total = 0.0
    while current < t_end:
        seg_end = min(current + step, t_end)
        lam = baseline.get_intensity(current)
        total += float(lam) * (seg_end - current).total_seconds() / 3600.0
        current = seg_end
    return total


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def make_run_id(prefix=None):
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}" if prefix else stamp

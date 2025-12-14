import pandas as pd

def temporal_train_test_split(times, train_ratio=0.8):
    t_start = times[0]
    duration = (times[-1] - t_start).total_seconds()
    split_point = t_start + pd.Timedelta(seconds=duration * train_ratio)
    train = [t for t in times if t <= split_point]
    test = [t for t in times if t > split_point]
    return train, test
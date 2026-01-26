import pandas as pd

EVENT_MAP = {"view": 0, "addtocart": 1, "transaction": 2}


def load_retailrocket(path, group="visitorid", max_sequences=None):
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    elif "time" in df.columns:
        df["ts"] = pd.to_datetime(df["time"])
    else:
        raise ValueError("Expected timestamp column in RetailRocket data")

    df = df[df["event"].isin(EVENT_MAP.keys())].copy()
    df["type_id"] = df["event"].map(EVENT_MAP)
    df = df.sort_values([group, "ts"])

    sequences = []
    for idx, (_, g) in enumerate(df.groupby(group)):
        times = g["ts"].tolist()
        types = g["type_id"].tolist()
        sequences.append({"id": g[group].iloc[0], "times": times, "types": types})
        if max_sequences is not None and len(sequences) >= max_sequences:
            break

    return sequences

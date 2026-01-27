import pandas as pd

EVENT_MAP = {"view": 0, "addtocart": 1, "transaction": 2}


def load_retailrocket(path, group="visitorid", max_sequences=None, min_events=None, require_transaction=False):
    usecols = ["event", group]
    if group != "timestamp":
        usecols.append("timestamp")
    df = pd.read_csv(path, usecols=list(dict.fromkeys(usecols)))
    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    elif "time" in df.columns:
        df["ts"] = pd.to_datetime(df["time"])
    else:
        raise ValueError("Expected timestamp column in RetailRocket data")

    df = df[df["event"].isin(EVENT_MAP.keys())].copy()
    df["type_id"] = df["event"].map(EVENT_MAP)
    df = df.sort_values([group, "ts"])

    if min_events is not None or require_transaction:
        counts = df.groupby(group, sort=False).size()
        eligible = counts.index
        if min_events is not None:
            eligible = counts[counts >= int(min_events)].index
        if require_transaction:
            counts_tx = df.loc[df["event"] == "transaction"].groupby(group, sort=False).size()
            eligible = eligible.intersection(counts_tx.index)
        if max_sequences is not None:
            eligible = counts.loc[eligible].sort_values(ascending=False).index[: int(max_sequences)]
        df = df[df[group].isin(eligible)]

    sequences = []
    for idx, (_, g) in enumerate(df.groupby(group)):
        times = g["ts"].tolist()
        types = g["type_id"].tolist()
        sequences.append({"id": g[group].iloc[0], "times": times, "types": types})
        if max_sequences is not None and len(sequences) >= max_sequences:
            break

    return sequences

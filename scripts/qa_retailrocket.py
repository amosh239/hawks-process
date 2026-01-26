#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, make_run_id


EVENTS = ["view", "addtocart", "transaction"]
RATIO_GRID = [0.5, 0.6, 0.7, 0.8]
MIN_EVENTS_GRID = [5, 10, 20, 30, 50, 80]

# Headless-safe matplotlib defaults (scripts should not depend on notebook backends).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="QA for RetailRocket events.csv (full file).")
    p.add_argument("--path", default="data/raw/retailrocket/events.csv", help="Path to RetailRocket events.csv")
    p.add_argument("--run-id", default=None, help="Run id (default: timestamp)")
    p.add_argument("--out-dir", default=None, help="Output dir (default: results/retailrocket/<run_id>/qa)")
    p.add_argument("--chunksize", type=int, default=500_000, help="CSV chunksize for streaming reads")
    p.add_argument("--write-per-visitor", action="store_true", help="Write per-visitor stats CSV (can be large)")
    return p.parse_args()


def _plot_hist(values, title, path, bins=50, logx=False, logy=False):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        plt.title(title + " (empty)")
    else:
        plt.hist(x, bins=bins, color="steelblue")
        plt.title(title)
    plt.xlabel(title)
    plt.ylabel("count")
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_heatmap(df_pivot, title, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    x = df_pivot.columns.astype(str).tolist()
    y = df_pivot.index.astype(str).tolist()
    z = df_pivot.values
    plt.imshow(z, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(x)), x)
    plt.yticks(range(len(y)), y)
    plt.xlabel("train_ratio")
    plt.ylabel("min_events")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            plt.text(j, i, str(int(z[i, j])), ha="center", va="center", fontsize=8, color="white")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()
    path = ROOT / args.path if not Path(args.path).is_absolute() else Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    run_id = args.run_id or make_run_id("retailrocket")
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "results" / "retailrocket" / run_id / "qa")
    ensure_dir(out_dir)

    usecols = ["timestamp", "visitorid", "event"]

    # Pass 1: per-visitor min/max timestamps + total event counts + global event counts
    visitor = {}  # vid -> [min_ts, max_ts, n_events]
    event_counts = {e: 0 for e in EVENTS}
    total_rows = 0
    global_min_ts = None
    global_max_ts = None

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunksize):
        chunk = chunk[chunk["event"].isin(EVENTS)]
        if chunk.empty:
            continue

        total_rows += len(chunk)
        ts_min = int(chunk["timestamp"].min())
        ts_max = int(chunk["timestamp"].max())
        global_min_ts = ts_min if global_min_ts is None else min(global_min_ts, ts_min)
        global_max_ts = ts_max if global_max_ts is None else max(global_max_ts, ts_max)

        vc = chunk["event"].value_counts()
        for e, c in vc.items():
            event_counts[e] = event_counts.get(e, 0) + int(c)

        g = chunk.groupby("visitorid")["timestamp"].agg(["min", "max", "size"])
        for vid, row in g.iterrows():
            vmin, vmax, n = int(row["min"]), int(row["max"]), int(row["size"])
            cur = visitor.get(int(vid))
            if cur is None:
                visitor[int(vid)] = [vmin, vmax, n]
            else:
                if vmin < cur[0]:
                    cur[0] = vmin
                if vmax > cur[1]:
                    cur[1] = vmax
                cur[2] += n

    vids = np.fromiter(visitor.keys(), dtype=np.int64, count=len(visitor))
    mins = np.empty(len(vids), dtype=np.int64)
    maxs = np.empty(len(vids), dtype=np.int64)
    n_events = np.empty(len(vids), dtype=np.int32)
    for i, vid in enumerate(vids):
        vmin, vmax, n = visitor[int(vid)]
        mins[i] = vmin
        maxs[i] = vmax
        n_events[i] = n

    # Precompute split thresholds for each ratio (per visitor).
    split_ts = {}
    span = (maxs - mins).astype(np.float64)
    for r in RATIO_GRID:
        split_ts[r] = (mins + (span * float(r))).astype(np.int64)

    # Pass 2: for each ratio, mark if visitor has a transaction in test window.
    vid_to_idx = {int(v): i for i, v in enumerate(vids)}
    tx_after = {r: np.zeros(len(vids), dtype=bool) for r in RATIO_GRID}

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunksize):
        chunk = chunk[chunk["event"] == "transaction"]
        if chunk.empty:
            continue

        vid_arr = chunk["visitorid"].astype(np.int64)
        ts_arr = chunk["timestamp"].astype(np.int64).to_numpy()
        idx = vid_arr.map(vid_to_idx)
        keep = idx.notna().to_numpy()
        if not np.any(keep):
            continue
        idx = idx.to_numpy()[keep].astype(np.int64)
        ts_arr = ts_arr[keep]

        for r in RATIO_GRID:
            hit = ts_arr > split_ts[r][idx]
            if np.any(hit):
                tx_after[r][idx[hit]] = True

    # Build grid reports
    rows = []
    for m in MIN_EVENTS_GRID:
        mask_m = n_events >= int(m)
        for r in RATIO_GRID:
            rows.append(
                {
                    "min_events": int(m),
                    "train_ratio": float(r),
                    "visitors_ge_min": int(mask_m.sum()),
                    "with_test_transaction": int(np.logical_and(mask_m, tx_after[r]).sum()),
                }
            )
    df_grid = pd.DataFrame(rows)
    df_grid.to_csv(out_dir / "filter_grid.csv", index=False)

    # Write summary JSON
    summary = {
        "path": str(path),
        "rows_total": int(total_rows),
        "unique_visitors": int(len(vids)),
        "event_counts": {k: int(v) for k, v in event_counts.items()},
        "min_timestamp_ms": int(global_min_ts) if global_min_ts is not None else None,
        "max_timestamp_ms": int(global_max_ts) if global_max_ts is not None else None,
        "time_min": pd.to_datetime(global_min_ts, unit="ms").isoformat() if global_min_ts is not None else None,
        "time_max": pd.to_datetime(global_max_ts, unit="ms").isoformat() if global_max_ts is not None else None,
        "chunksize": int(args.chunksize),
    }
    (out_dir / "qa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Optional per-visitor stats (small columns only)
    if args.write_per_visitor:
        span_hours = (maxs - mins).astype(np.float64) / (1000.0 * 3600.0)
        pv = pd.DataFrame({"visitorid": vids, "n_events": n_events, "span_hours": span_hours})
        pv.to_csv(out_dir / "per_visitor_stats.csv", index=False)

    # Plots
    _plot_hist(n_events, "events per visitor", out_dir / "hist_n_events.png", bins=60, logx=True, logy=True)
    _plot_hist((maxs - mins) / (1000.0 * 3600.0), "span_hours per visitor", out_dir / "hist_span_hours.png", bins=60, logx=True, logy=True)
    _plot_heatmap(
        df_grid.pivot(index="min_events", columns="train_ratio", values="with_test_transaction"),
        "visitors with >=min_events and a test transaction",
        out_dir / "heatmap_with_test_transaction.png",
    )
    _plot_heatmap(
        df_grid.pivot(index="min_events", columns="train_ratio", values="visitors_ge_min"),
        "visitors with >=min_events (upper bound)",
        out_dir / "heatmap_visitors_ge_min.png",
    )

    # Console summary (so you don't have to open files just to sanity-check).
    print("Wrote QA to:", out_dir)
    print("Summary:", out_dir / "qa_summary.json")
    print("Grid:", out_dir / "filter_grid.csv")
    print("\nEvent counts:", summary["event_counts"])
    print("Unique visitors:", summary["unique_visitors"])
    print("Time range:", summary["time_min"], "->", summary["time_max"])
    print("\nwith_test_transaction (rows=min_events, cols=train_ratio):")
    print(df_grid.pivot(index="min_events", columns="train_ratio", values="with_test_transaction").to_string())


if __name__ == "__main__":
    main()

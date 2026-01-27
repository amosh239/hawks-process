#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from src.datasets import load_retailrocket
from src.experiments import run_purchase_experiment
from src.utils import make_run_id, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-data Hawkes experiment.")
    parser.add_argument("--dataset", choices=["retailrocket"], default="retailrocket")
    parser.add_argument("--path", default="data/raw/retailrocket/events.csv", help="Path to events.csv")
    parser.add_argument("--min-events", type=int, default=30)
    parser.add_argument("--min-train-events", type=int, default=20)
    parser.add_argument("--min-purchases-train", type=int, default=2)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Limit to a tiny subset")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.path).exists():
        raise FileNotFoundError(f"Missing file: {args.path}")
    max_seqs = args.max_seqs
    if args.smoke:
        # Pick enough sequences so we reliably get some purchases, but keep it fast.
        max_seqs = 10
        args.min_events = 20
        args.min_train_events = 15
        args.min_purchases_train = 1
        args.train_ratio = 0.7

    if args.dataset == "retailrocket":
        sequences = load_retailrocket(
            args.path,
            max_sequences=max_seqs,
            min_events=args.min_events,
            require_transaction=True,
        )
    else:
        raise ValueError("Unsupported dataset")

    run_id = make_run_id(args.dataset)
    out_dir = Path("results") / args.dataset / run_id
    ensure_dir(out_dir)

    df = run_purchase_experiment(
        sequences,
        results_dir=out_dir,
        train_ratio=args.train_ratio,
        min_events=args.min_events,
        min_train_events=args.min_train_events,
        min_purchases_train=args.min_purchases_train,
        max_sequences=max_seqs,
    )
    print(f"Saved results to {out_dir / 'summary.csv'} with {len(df)} rows.")


if __name__ == "__main__":
    main()

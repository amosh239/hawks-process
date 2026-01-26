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

from src.datasets import generate_synthetic_sequences
from src.experiments import run_univariate_experiment
from src.utils import make_run_id, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic Hawkes experiment.")
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Tiny run for CI/checks")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.n_users = 5
        args.min_events = 5
        args.max_seqs = 5
        args.train_ratio = 0.7

    sequences, _ = generate_synthetic_sequences(n_users=args.n_users, seed=args.seed)
    run_id = make_run_id("synthetic")
    out_dir = Path("results") / "synthetic" / run_id
    ensure_dir(out_dir)

    df = run_univariate_experiment(
        sequences,
        results_dir=out_dir,
        train_ratio=args.train_ratio,
        min_events=args.min_events,
        max_sequences=args.max_seqs,
    )
    print(f"Saved results to {out_dir / 'summary.csv'} with {len(df)} rows.")


if __name__ == "__main__":
    main()

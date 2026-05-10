from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mpl_config = ROOT / ".mplconfig"
xdg_cache = ROOT / ".cache"
mpl_config.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

from src.diploma_baselines import run_global_poisson_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global Poisson baseline on daily user panel")
    parser.add_argument(
        "--data-path",
        default="data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        help="Path to daily panel csv",
    )
    parser.add_argument("--target-col", default="to_ord", help="Target count column")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Share of calendar days in train")
    parser.add_argument("--analysis-start", default="2025-01-15", help="Requested analysis window start date")
    parser.add_argument("--analysis-end", default="2025-09-30", help="Requested analysis window end date")
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/poisson_baseline",
        help="Directory for figures and summary artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_global_poisson_experiment(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        target_col=args.target_col,
        train_ratio=args.train_ratio,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

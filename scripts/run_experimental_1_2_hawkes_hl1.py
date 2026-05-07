from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mpl_config = ROOT / ".mplconfig"
xdg_cache = ROOT / ".cache"
mpl_config.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

from src.diploma_baselines.pipeline import run_hawkes_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short-memory Hawkes with only half-life=1 on current diploma split"
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        help="Path to daily grid csv",
    )
    parser.add_argument("--target-col", default="to_ord", help="Target count column")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Share of calendar days in train")
    parser.add_argument("--analysis-start", default="2025-01-15", help="Requested analysis window start date")
    parser.add_argument("--analysis-end", default="2025-09-30", help="Requested analysis window end date")
    parser.add_argument("--window-size", type=int, default=7, help="Rolling window for chapter-4 baseline")
    parser.add_argument("--alpha-l2", type=float, default=1e-4, help="L2 regularization for pooled Hawkes alpha")
    parser.add_argument(
        "--scale-l2",
        type=float,
        default=10.0,
        help="Quadratic regularization that keeps the learned baseline scale near 1",
    )
    parser.add_argument("--scale-init", type=float, default=1.0, help="Initial value for learned baseline scale")
    parser.add_argument("--max-iter", type=int, default=300, help="Optimizer max iterations")
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/experimental_1_2_hawkes_hl1",
        help="Directory for short-memory Hawkes artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_hawkes_experiment(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        target_col=args.target_col,
        train_ratio=args.train_ratio,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
        window_size=args.window_size,
        half_lives=(1.0,),
        alpha_l2=args.alpha_l2,
        learn_base_scale=True,
        scale_l2=args.scale_l2,
        scale_init=args.scale_init,
        max_iter=args.max_iter,
        model_label="Short-memory Hawkes (hl=1)",
        model_slug="experimental_1_2_hawkes_hl1",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

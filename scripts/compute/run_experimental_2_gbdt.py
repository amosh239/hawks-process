from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

from src.diploma_experimental import run_experimental_2_gbdt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experimental day-level GBDT on current diploma split")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--max-users", type=int, default=None, help="Optional deterministic cap on number of users")
    parser.add_argument(
        "--source-features",
        default="search,cat,searches,has_search_to_cart,has_search_to_ord,has_cat_to_cart,has_cat_to_ord,search_to_cart,search_to_ord,cat_to_cart,cat_to_ord,to_cart,to_ord,gmv",
        help="Comma-separated source features for history engineering",
    )
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/experimental_2_gbdt",
        help="Directory for experimental GBDT artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_features = [x.strip() for x in str(args.source_features).split(",") if x.strip()]
    summary = run_experimental_2_gbdt(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        target_col=args.target_col,
        train_ratio=args.train_ratio,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
        window_size=args.window_size,
        seed=args.seed,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        min_samples_leaf=args.min_samples_leaf,
        source_features=source_features,
        max_users=args.max_users,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

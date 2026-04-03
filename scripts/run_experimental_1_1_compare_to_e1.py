from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

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

from src.diploma_baselines.plots import plot_delta_ll_vs_test_purchases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare E1.1 user-scale Hawkes against current E1 Hawkes baseline")
    parser.add_argument(
        "--e1-user-ll",
        default="diploma/reports/experimental_1_hawkes/user_ll_scores.csv",
        help="Path to E1 user-level LL csv",
    )
    parser.add_argument(
        "--e11-user-ll",
        default="diploma/reports/experimental_1_1_hawkes/user_ll_scores.csv",
        help="Path to E1.1 user-level LL csv",
    )
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/experimental_1_1_hawkes",
        help="Directory for compare artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    e1 = pd.read_csv(args.e1_user_ll)
    e11 = pd.read_csv(args.e11_user_ll)
    merged = e1.merge(
        e11[["user_id", "ll_user_scale_hawkes"]],
        on="user_id",
        how="outer",
    ).fillna(0.0)
    merged.to_csv(output_dir / "user_ll_scores_vs_e1.csv", index=False)

    delta = merged["ll_user_scale_hawkes"] - merged["ll_experimental_1_hawkes"]
    summary = {
        "users": int(len(merged)),
        "share_new_better": float((delta > 0).mean()),
        "mean_delta_ll": float(delta.mean()),
        "median_delta_ll": float(delta.median()),
        "q10_delta_ll": float(delta.quantile(0.1)),
        "q90_delta_ll": float(delta.quantile(0.9)),
    }
    with open(output_dir / "compare_to_e1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_delta_ll_vs_test_purchases(
        user_ll_df=merged,
        prev_col="ll_experimental_1_hawkes",
        new_col="ll_user_scale_hawkes",
        purchases_col="test_purchases",
        out_path=output_dir / "delta_ll_vs_test_purchases_e1_to_e11.png",
        title="Per-user delta LL vs test purchases: E1 -> E1.1",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

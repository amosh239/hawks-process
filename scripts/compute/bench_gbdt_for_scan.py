"""Quick benchmark: time GBDT (feature build + fit + predict) for a small
and a large interval matching the chapter-11 scan grid. Used to estimate
how long adding GBDT to the train-length scan would take.
"""

from __future__ import annotations

import os
import sys
import time
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

import numpy as np
import pandas as pd

from src.diploma_baselines.data import load_daily_grid
from src.diploma_experimental.gbdt import (
    SOURCE_FEATURES,
    build_feature_tables,
    fit_global_poisson_gbdt,
)


GLOBAL_START = pd.Timestamp("2025-01-15")
DATA_PATH = "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv"


def time_one(full_df: pd.DataFrame, n_days: int) -> dict:
    interval_start = GLOBAL_START
    train_len = (n_days // 3) * 2
    train_end = interval_start + pd.Timedelta(days=train_len - 1)
    interval_end = interval_start + pd.Timedelta(days=n_days - 1)

    t0 = time.time()
    ft = build_feature_tables(
        full_df=full_df,
        analysis_start=interval_start,
        analysis_end=interval_end,
        split_date=train_end,
        target_col="to_ord",
        source_features=SOURCE_FEATURES,
    )
    t_feat = time.time() - t0

    t0 = time.time()
    model = fit_global_poisson_gbdt(ft, seed=42, max_depth=5,
                                    learning_rate=0.05, max_iter=200,
                                    min_samples_leaf=40)
    t_fit = time.time() - t0

    t0 = time.time()
    _ = model.predict(ft.x_test)
    t_pred = time.time() - t0

    return {
        "n_days": n_days,
        "train_rows": int(len(ft.y_train)),
        "test_rows": int(len(ft.y_test)),
        "t_feat_s": t_feat,
        "t_fit_s": t_fit,
        "t_pred_s": t_pred,
        "t_total_s": t_feat + t_fit + t_pred,
    }


def main() -> None:
    print("Loading data...")
    cols = list(set(["to_ord", *SOURCE_FEATURES]))
    # build_feature_tables expects all SOURCE_FEATURES in the dataframe
    full_df = load_daily_grid(DATA_PATH, value_cols=cols)
    print(f"  {len(full_df):,} rows, users={full_df['user_id'].nunique()}")

    for n in [15, 63, 198]:
        r = time_one(full_df, n)
        print(
            f"n={r['n_days']:>3}d  train={r['train_rows']:>7,}  test={r['test_rows']:>7,}  "
            f"feat={r['t_feat_s']:6.2f}s  fit={r['t_fit_s']:6.2f}s  pred={r['t_pred_s']:5.2f}s  "
            f"total={r['t_total_s']:6.2f}s"
        )


if __name__ == "__main__":
    main()

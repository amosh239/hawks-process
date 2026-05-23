"""Chapter 17 (script tag `ch25`): per-channel boosting baseline.

For each of 3 channels (`searches`, `to_cart`, `to_ord`) as target, train a
gradient-boosting regressor (HistGradientBoostingRegressor with Poisson loss)
on the full feature-engineering set used by experimental GBDT in chapter 9
(~132 features: dow, sum3/7, mean7, active_days7, exp3, recency, EMA, funnel
ratios — see SOURCE_FEATURES in src.diploma_experimental.gbdt).

Same train/test split as chapter 15:
  Train: 2025-01-15 .. 2025-08-09 (~207d)
  Test:  2025-08-10 .. 2025-09-30 (~52d)

Outputs (under `diploma/reports/25_cross_channel_gbdt/`):
  - summary.json
  - per_channel_metrics.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.diploma_baselines.data import filter_date_range, load_daily_grid
from src.diploma_baselines.metrics import evaluate_count_forecast
from src.diploma_experimental.gbdt import SOURCE_FEATURES, build_feature_panel


CHANNELS = ("searches", "to_cart", "to_ord")

ANALYSIS_START = pd.Timestamp("2025-01-15")
ANALYSIS_END = pd.Timestamp("2025-09-30")
SPLIT_DATE = pd.Timestamp("2025-08-09")  # train ≤ this, test > this

PANEL_PATH = "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv"
OUT_DIR = Path("diploma/reports/25_cross_channel_gbdt")

GBDT_PARAMS = dict(
    loss="poisson",
    max_iter=400,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=200,
    l2_regularization=1.0,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading panel...")
    full_df = load_daily_grid(PANEL_PATH, value_cols=SOURCE_FEATURES)
    print(f"  loaded {len(full_df):,} rows")

    print("\nBuilding full feature panel (this is the slow part)...")
    t0 = time.time()
    # target_col here only controls index_df['target']; features are independent of target choice.
    x_panel, index_df, feature_names = build_feature_panel(
        full_df=full_df,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        target_col="to_ord",
        source_features=SOURCE_FEATURES,
    )
    print(f"  built {len(feature_names)} features for {len(index_df):,} rows in {time.time()-t0:.1f}s")

    # Train/test mask: train if event_date <= SPLIT_DATE
    is_train = (index_df["event_date"] <= SPLIT_DATE).to_numpy()
    x_train = x_panel[is_train]
    x_test = x_panel[~is_train]
    train_index = index_df[is_train].reset_index(drop=True)
    test_index = index_df[~is_train].reset_index(drop=True)
    print(f"  train rows: {len(train_index):,}; test rows: {len(test_index):,}")

    # Lookup per-channel targets via merge on (user_id, event_date)
    print("\nFetching per-channel targets via merge...")
    needed = full_df[["user_id", "event_date", *CHANNELS]].copy()
    train_y_df = train_index.merge(needed, on=["user_id", "event_date"], how="left")
    test_y_df = test_index.merge(needed, on=["user_id", "event_date"], how="left")
    assert not train_y_df[list(CHANNELS)].isna().any().any(), "missing channel target rows in train"
    assert not test_y_df[list(CHANNELS)].isna().any().any(), "missing channel target rows in test"

    per_channel_rows = []
    for target in CHANNELS:
        print(f"\n=== target = {target} ===")
        y_train = train_y_df[target].to_numpy(dtype=float)
        y_test = test_y_df[target].to_numpy(dtype=float)
        print(f"  train mean y = {y_train.mean():.4f}, test mean y = {y_test.mean():.4f}")

        t0 = time.time()
        model = HistGradientBoostingRegressor(**GBDT_PARAMS)
        model.fit(x_train, y_train)
        elapsed = time.time() - t0
        print(f"  fit done in {elapsed:.1f}s; n_iter_={model.n_iter_}")

        lam_test = np.clip(model.predict(x_test), 1e-8, None)
        metrics = evaluate_count_forecast(y_test, lam_test)
        nll = float(metrics["mean_poisson_nll"])
        print(f"  test NLL = {nll:.4f}")

        per_channel_rows.append({
            "target": target,
            "test_nll": nll,
            "n_iter": int(model.n_iter_),
            "fit_seconds": float(elapsed),
        })

    pd.DataFrame(per_channel_rows).to_csv(OUT_DIR / "per_channel_metrics.csv", index=False)

    summary = {
        "channels": list(CHANNELS),
        "analysis_window": [str(ANALYSIS_START.date()), str(ANALYSIS_END.date())],
        "split_date": str(SPLIT_DATE.date()),
        "n_features": int(len(feature_names)),
        "feature_names_count": int(len(feature_names)),
        "source_features": list(SOURCE_FEATURES),
        "gbdt_params": GBDT_PARAMS,
        "per_channel_metrics": per_channel_rows,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()

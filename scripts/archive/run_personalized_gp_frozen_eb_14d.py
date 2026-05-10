"""Personalized Gamma-Poisson with FROZEN EB hyperparameters on 14d blocks.

The standard Personalized GP in chapter 4 / chapter 10 re-fits the Gamma
hyperparameters (α, β) by marginal MLE on each train block. On 14d blocks
this gives noisy hyperparameters around (0.34, 0.34), which then drive the
posterior shrinkage.

This script tests whether the *form* of Gamma-Poisson is the problem, or only
the marginal MLE estimation. We FREEZE (α, β) at the values that marginal MLE
gives on the 207d long train (chapter 6 main split):

    α = 0.8774, β = 0.8808  (from reports/experimental_1_hawkes/summary.json)

Then on each 14d block we:
  1. aggregate per-user (Y_u, E_u) on 14d train,
  2. compute posterior mean μ_u = (α + Y_u) / (β + E_u) — with FROZEN α, β,
  3. predict on 7d test.

Comparison set on each block:
  - re-fit EB (current Personalized GP, ch.10)        — fits α, β from 14d
  - frozen EB (this script)                            — uses α, β from 207d
  - standalone L2 (γ=1, ch.15)                         — Gaussian L2 prior

If frozen EB ≈ standalone L2 → the bug was in marginal MLE on small samples.
If frozen EB ≈ re-fit EB → the Gamma-Poisson form itself struggles on 14d.
"""

from __future__ import annotations

import os
import sys
import time
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

import numpy as np
import pandas as pd

from src.diploma_baselines.data import load_daily_grid
from src.diploma_baselines.models import GlobalRollingSeasonalPoissonModel


TARGET_COL = "to_ord"
WINDOW_SIZE = 7

CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14

# Frozen EB hyperparameters from chapter 6 / chapter 4 (207d marginal MLE)
ALPHA_LONG = 0.8774
BETA_LONG = 0.8808
PRIOR_MEAN_LONG = ALPHA_LONG / BETA_LONG  # ≈ 0.996

OUTPUT_DIR = Path("diploma/reports/blockwise_cv")


from src.diploma_baselines.metrics import evaluate_count_forecast


def standard_poisson_nll(y: np.ndarray, lam: np.ndarray) -> float:
    """Full Poisson NLL via evaluate_count_forecast (includes log(y!) term)."""
    return float(evaluate_count_forecast(np.asarray(y, dtype=float), np.asarray(lam, dtype=float))["mean_poisson_nll"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=[TARGET_COL],
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    blocks = []
    cursor = CV_GLOBAL_START
    idx = 0
    while True:
        block_start = cursor
        block_end = cursor + pd.Timedelta(days=BLOCK_LEN - 1)
        if block_end > CV_GLOBAL_END:
            break
        train_end = block_start + pd.Timedelta(days=TRAIN_LEN - 1)
        blocks.append({
            "block_idx": idx,
            "block_start": block_start,
            "train_end": train_end,
            "test_start": train_end + pd.Timedelta(days=1),
            "test_end": block_end,
        })
        idx += 1
        cursor = block_end + pd.Timedelta(days=1)

    print(f"\n=== Personalized GP with FROZEN EB (α={ALPHA_LONG}, β={BETA_LONG}) on each 14d block ===\n")
    print(f"  prior_mean (frozen) = {PRIOR_MEAN_LONG:.4f}\n")

    rows = []
    for block in blocks:
        block_start = block["block_start"]
        train_end = block["train_end"]
        test_start = block["test_start"]
        test_end = block["test_end"]

        block_train_df = full_df.loc[
            (full_df["event_date"] >= block_start) & (full_df["event_date"] <= train_end)
        ].copy()
        block_test_df = full_df.loc[
            (full_df["event_date"] >= test_start) & (full_df["event_date"] <= test_end)
        ].copy()

        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs_block = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean, daily_mean_full
        )
        base_train = rs_block.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)
        base_test = rs_block.predict_for_dates(block_test_df["event_date"]).to_numpy(dtype=float)

        # Per-user (Y_u, E_u) on 14d train
        train_df_aug = block_train_df.copy()
        train_df_aug["base_lambda"] = base_train
        per_user = train_df_aug.groupby("user_id").agg(
            y_sum=(TARGET_COL, "sum"),
            exposure=("base_lambda", "sum"),
        )

        # Posterior mean with FROZEN α, β
        mu_post = (ALPHA_LONG + per_user["y_sum"]) / (BETA_LONG + per_user["exposure"])

        # Predict on test
        test_uids = block_test_df["user_id"].to_numpy()
        mu_for_test = pd.Series(test_uids).map(mu_post).fillna(PRIOR_MEAN_LONG).to_numpy(dtype=float)
        lam_test = np.clip(mu_for_test * base_test, 1e-8, None)
        y_test = block_test_df[TARGET_COL].to_numpy(dtype=float)
        nll = standard_poisson_nll(y_test, lam_test)

        rows.append({
            "block_idx": block["block_idx"],
            "block_label": f"{block_start.date()}..{block['block_start'] + pd.Timedelta(days=BLOCK_LEN - 1):%Y-%m-%d}",
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "test_rows": int(len(block_test_df)),
            "Personalized GP frozen-EB": nll,
            "frozen_alpha": ALPHA_LONG,
            "frozen_beta": BETA_LONG,
            "mu_post_mean": float(mu_post.mean()),
            "mu_post_median": float(mu_post.median()),
        })

        print(
            f"  block {block['block_idx'] + 1:>2}/13 {block_start.date()}..{train_end.date()}  "
            f"NLL={nll:.4f}  μ_post: mean={mu_post.mean():.3f}  median={mu_post.median():.3f}"
        )

    out_path = OUTPUT_DIR / "personalized_gp_frozen_eb_14d.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nMean NLL across 13 blocks (frozen-EB) = {df['Personalized GP frozen-EB'].mean():.4f}")
    print(f"Saved to {out_path}")

    # Compare with re-fit EB and standalone L2
    refit_eb_csv = Path("diploma/reports/blockwise_cv/cv_results.csv")
    l2_csv = Path("diploma/reports/joint_lambda_alpha/personalized_l2_standalone_14d.csv")

    refit_eb_df = pd.read_csv(refit_eb_csv)[["block_idx", "Personalized Gamma-Poisson"]].rename(
        columns={"Personalized Gamma-Poisson": "Personalized GP refit-EB"}
    )
    l2_df = pd.read_csv(l2_csv)[["block_idx", "Personalized L2 standalone"]]

    summary = df[["block_idx", "Personalized GP frozen-EB"]].merge(
        refit_eb_df, on="block_idx"
    ).merge(l2_df, on="block_idx")

    print("\n=== Per-block NLL comparison ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"\n=== Mean across 13 blocks ===\n"
        f"  Personalized GP refit-EB (current ch.10):    {summary['Personalized GP refit-EB'].mean():.4f}\n"
        f"  Personalized GP frozen-EB (this experiment): {summary['Personalized GP frozen-EB'].mean():.4f}\n"
        f"  Personalized L2 standalone (γ=1):            {summary['Personalized L2 standalone'].mean():.4f}"
    )

    summary.to_csv(OUTPUT_DIR / "personalized_three_baselines_compare.csv", index=False)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

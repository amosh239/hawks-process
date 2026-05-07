"""Test the asymmetry hypothesis from chapter 13.

For each of the 13 three-week CV blocks, on the 7d TEST window run TWO setups:

  Setup F (FULL refit on test, symmetric to train-fit):
    EB scaler .fit(test_df) → pers_test_in_sample
    Hawkes .fit(states_test, y_test, base = pers_test_in_sample)
    [in-sample EB on test, just like train fit uses in-sample EB on train]

  Setup H (HAWKES-only on test, asymmetric, like old chapter 13):
    EB scaler .fit(train_df) → pers_test_out_of_sample
    Hawkes .fit(states_test, y_test, base = pers_test_out_of_sample)
    [out-of-sample EB applied to test]

Hypothesis: setup F degenerates to (c=1, alpha=0) just like train fit does
(because in-sample EB residual is ~0 by construction). If true, the chapter 13
"test-loss optimum is in a different place" finding was an artifact of the
out-of-sample baseline, not a real test-loss vs train-loss asymmetry.
"""

from __future__ import annotations

import json
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diploma_baselines.data import load_daily_grid
from src.diploma_baselines.models import (
    FEATURE_NAMES,
    GlobalRollingSeasonalPoissonModel,
    PersonalizedGammaPoissonScaler,
    build_basis_states,
    fit_pooled_additive_multi_kernel_hawkes,
)


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14

OUTPUT_DIR = Path("diploma/reports/hawkes_test_fit_symmetric")


def standard_poisson_nll(y: np.ndarray, lam: np.ndarray) -> float:
    lam = np.clip(lam, 1e-8, None)
    return float(np.mean(lam - y * np.log(lam)))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full_target = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()

    beta = np.log(2.0) / np.asarray(HALF_LIVES, dtype=float)
    print("\nBuilding Hawkes states from full history...")
    user_states: dict[int, dict] = {}
    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        user_states[int(user_id)] = {
            "states_full": states_full,
            "dates": full_user["event_date"].to_numpy(dtype="datetime64[ns]"),
        }
    print(f"  states for {len(user_states):,} users")

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

    print("\n=== Two setups on the 7d TEST window of each block ===")
    print("  F: full refit on test (in-sample EB on test) → Hawkes")
    print("  H: EB from train (out-of-sample EB) → Hawkes on test (= old ch.13)\n")

    HAWKES_KW = dict(
        half_lives=HALF_LIVES,
        feature_names=HAWKES_FEATURES,
        alpha_l2=0.0,
        learn_base_scale=True,
        scale_l2=0.0,
        scale_init=1.0,
        max_iter=500,
    )

    results = []
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

        # === RS baseline always fit on the same train block (it's a global thing,
        # we don't refit RS on 7d which would be too noisy)
        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs_block = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean, daily_mean_full_target
        )
        base_train = rs_block.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)
        base_test = rs_block.predict_for_dates(block_test_df["event_date"]).to_numpy(dtype=float)

        # === Setup H: EB fit on train, applied to test (out-of-sample) ===
        scaler_train = PersonalizedGammaPoissonScaler().fit(
            block_train_df["user_id"].to_numpy(),
            block_train_df[TARGET_COL].to_numpy(),
            base_train,
        )
        pers_test_oos = scaler_train.predict(
            block_test_df["user_id"].to_numpy(), base_test, method="posterior_mean"
        )

        # === Setup F: EB fit on test (in-sample EB on test) ===
        scaler_test = PersonalizedGammaPoissonScaler().fit(
            block_test_df["user_id"].to_numpy(),
            block_test_df[TARGET_COL].to_numpy(),
            base_test,
        )
        pers_test_is = scaler_test.predict(
            block_test_df["user_id"].to_numpy(), base_test, method="posterior_mean"
        )

        # Test states
        test_dates = block_test_df["event_date"].to_numpy(dtype="datetime64[ns]")
        all_states = np.zeros((len(block_test_df), 10), dtype=np.float32)
        state_blocks_oos = []
        state_blocks_is = []
        y_blocks = []
        base_blocks_oos = []
        base_blocks_is = []
        for uid, idx_in_block in block_test_df.groupby("user_id", sort=False).indices.items():
            info = user_states[int(uid)]
            full_dates = info["dates"]
            wanted_dates = test_dates[idx_in_block]
            full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}
            rows_in_full = np.array(
                [full_to_idx[pd.Timestamp(d).normalize()] for d in wanted_dates], dtype=int
            )
            user_states_test = info["states_full"][rows_in_full]
            all_states[idx_in_block] = user_states_test
            y_user = block_test_df[TARGET_COL].to_numpy(dtype=float)[idx_in_block]
            state_blocks_oos.append(user_states_test)
            state_blocks_is.append(user_states_test)
            y_blocks.append(y_user)
            base_blocks_oos.append(pers_test_oos[idx_in_block])
            base_blocks_is.append(pers_test_is[idx_in_block])

        y_test = block_test_df[TARGET_COL].to_numpy(dtype=float)

        # Setup H: Hawkes fit on test with out-of-sample EB
        fit_h = fit_pooled_additive_multi_kernel_hawkes(
            state_blocks=state_blocks_oos,
            y_blocks=y_blocks,
            base_blocks=base_blocks_oos,
            **HAWKES_KW,
        )
        lam_h = fit_h.base_scale * pers_test_oos + all_states.astype(float) @ fit_h.alpha
        nll_h = standard_poisson_nll(y_test, lam_h)

        # Setup F: Hawkes fit on test with in-sample EB
        fit_f = fit_pooled_additive_multi_kernel_hawkes(
            state_blocks=state_blocks_is,
            y_blocks=y_blocks,
            base_blocks=base_blocks_is,
            **HAWKES_KW,
        )
        lam_f = fit_f.base_scale * pers_test_is + all_states.astype(float) @ fit_f.alpha
        nll_f = standard_poisson_nll(y_test, lam_f)

        # Reference: in-sample EB alone, no Hawkes
        nll_pers_is = standard_poisson_nll(y_test, pers_test_is)
        # Reference: out-of-sample EB alone, no Hawkes
        nll_pers_oos = standard_poisson_nll(y_test, pers_test_oos)

        row = {
            "block_idx": block["block_idx"],
            "label": f"{block_start.date()}..{train_end.date()}",
            "n_test": int(len(block_test_df)),
            "nll_pers_oos_only": nll_pers_oos,
            "nll_pers_is_only": nll_pers_is,
            "nll_setup_H_oos_eb_plus_hawkes": nll_h,
            "nll_setup_F_is_eb_plus_hawkes": nll_f,
            "H_c": float(fit_h.base_scale),
            "H_alpha_norm": float(np.linalg.norm(fit_h.alpha)),
            "F_c": float(fit_f.base_scale),
            "F_alpha_norm": float(np.linalg.norm(fit_f.alpha)),
        }
        results.append(row)

        print(
            f"  block {block['block_idx'] + 1:>2}/13 "
            f"H: c={fit_h.base_scale:.3f} ||α||={row['H_alpha_norm']:.4f} NLL={nll_h:.4f}  |  "
            f"F: c={fit_f.base_scale:.3f} ||α||={row['F_alpha_norm']:.4f} NLL={nll_f:.4f}"
        )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "test_fit_symmetric_per_block.csv", index=False)

    summary = {
        "n_blocks": int(len(df)),
        "mean_H_c": float(df["H_c"].mean()),
        "mean_H_alpha_norm": float(df["H_alpha_norm"].mean()),
        "mean_F_c": float(df["F_c"].mean()),
        "mean_F_alpha_norm": float(df["F_alpha_norm"].mean()),
        "mean_nll_H": float(df["nll_setup_H_oos_eb_plus_hawkes"].mean()),
        "mean_nll_F": float(df["nll_setup_F_is_eb_plus_hawkes"].mean()),
        "mean_nll_pers_oos_only": float(df["nll_pers_oos_only"].mean()),
        "mean_nll_pers_is_only": float(df["nll_pers_is_only"].mean()),
        "F_degenerate_count": int((df["F_alpha_norm"] < 1e-4).sum()),
        "H_degenerate_count": int((df["H_alpha_norm"] < 1e-4).sum()),
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary across 13 blocks:")
    print(f"  Setup H (out-of-sample EB + Hawkes):  mean c={summary['mean_H_c']:.3f}  mean ||α||={summary['mean_H_alpha_norm']:.4f}  mean NLL={summary['mean_nll_H']:.4f}  degenerate {summary['H_degenerate_count']}/13")
    print(f"  Setup F (in-sample EB + Hawkes):      mean c={summary['mean_F_c']:.3f}  mean ||α||={summary['mean_F_alpha_norm']:.4f}  mean NLL={summary['mean_nll_F']:.4f}  degenerate {summary['F_degenerate_count']}/13")
    print(f"  reference: out-of-sample EB only NLL = {summary['mean_nll_pers_oos_only']:.4f}")
    print(f"  reference: in-sample EB only NLL     = {summary['mean_nll_pers_is_only']:.4f}")

    # === Plot: ||alpha||_2 by setup, by block ===
    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    x_idx = np.arange(len(df))
    ax.bar(x_idx - 0.20, df["H_alpha_norm"].to_numpy(dtype=float), 0.40, label="Setup H: train-EB → Hawkes-on-test", color="#2E5EAA", edgecolor="white")
    ax.bar(x_idx + 0.20, df["F_alpha_norm"].to_numpy(dtype=float), 0.40, label="Setup F: test-EB → Hawkes-on-test", color="#A0522D", edgecolor="white")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"B{i + 1}" for i in range(len(df))])
    ax.set_ylabel("Fitted ||α||_2")
    ax.set_title("||α||_2 from Hawkes on 7d test: out-of-sample baseline (H) vs in-sample baseline (F)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "alpha_norm_per_block.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

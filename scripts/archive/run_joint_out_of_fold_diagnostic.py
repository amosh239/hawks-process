"""Out-of-fold diagnostic for joint Hawkes (chapter 14, diagnostic 3).

For each pair (B_k → B_{k+1}):
  1. Fit joint (λ_u, α) on the 14d train of B_k (γ=1, same as ch. 14).
  2. Apply (λ_u, α) to the 7d TEST of B_{k+1}, using RS baseline that was fit
     on B_k's train (so EVERYTHING comes from B_k).
  3. Compare the resulting test NLL with:
       - in-fold:  joint of B_k applied to B_k's own test;
       - in-fold:  joint of B_{k+1} applied to B_{k+1}'s own test.

If out-of-fold NLL is close to in-fold, the joint solution generalizes.
If it blows up, joint is overfitting to the block.
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
    build_basis_states,
)

from scripts.run_joint_lambda_alpha_fit import fit_joint  # type: ignore


TARGET_COL = "to_ord"
HAWKES_FEATURES = tuple(FEATURE_NAMES)
HALF_LIVES = (1.0, 3.0)
WINDOW_SIZE = 7

CV_GLOBAL_START = pd.Timestamp("2025-01-15")
CV_GLOBAL_END = pd.Timestamp("2025-10-31")
BLOCK_LEN = 21
TRAIN_LEN = 14

OUTPUT_DIR = Path("diploma/reports/joint_lambda_alpha/out_of_fold")


from src.diploma_baselines.metrics import evaluate_count_forecast


def standard_poisson_nll(y: np.ndarray, lam: np.ndarray) -> float:
    """Full Poisson NLL via evaluate_count_forecast (includes log(y!) term)."""
    return float(evaluate_count_forecast(np.asarray(y, dtype=float), np.asarray(lam, dtype=float))["mean_poisson_nll"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cols = list(dict.fromkeys([TARGET_COL, *HAWKES_FEATURES]))
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=cols,
    )
    print(f"  loaded {len(full_df):,} rows")

    daily_mean_full = full_df.groupby("event_date")[TARGET_COL].mean().sort_index()
    beta = np.log(2.0) / np.asarray(HALF_LIVES, dtype=float)

    print("\nBuilding Hawkes states from full history...")
    user_states_per_id: dict[int, dict] = {}
    for user_id, full_user in full_df.groupby("user_id", sort=False):
        x_full = full_user.loc[:, list(HAWKES_FEATURES)].to_numpy(dtype=float)
        states_full = build_basis_states(x_full, beta).reshape(len(full_user), -1).astype(np.float32)
        user_states_per_id[int(user_id)] = {
            "states_full": states_full,
            "dates": full_user["event_date"].to_numpy(dtype="datetime64[ns]"),
        }
    n_alpha = next(iter(user_states_per_id.values()))["states_full"].shape[1]

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

    def gather_states_for(df: pd.DataFrame) -> np.ndarray:
        dates = df["event_date"].to_numpy(dtype="datetime64[ns]")
        out = np.zeros((len(df), n_alpha), dtype=np.float32)
        for uid, idx_in_block in df.groupby("user_id", sort=False).indices.items():
            info = user_states_per_id[int(uid)]
            full_dates = info["dates"]
            wanted_dates = dates[idx_in_block]
            full_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(full_dates)}
            rows_in_full = np.array(
                [full_to_idx[pd.Timestamp(d).normalize()] for d in wanted_dates], dtype=int
            )
            out[idx_in_block] = info["states_full"][rows_in_full]
        return out

    # === Step 1: fit joint on each block once ===
    print(f"\n=== Fitting joint (λ_u, α) on each of {len(blocks)} blocks ===")
    fits: dict[int, dict] = {}
    for block in blocks:
        block_start = block["block_start"]
        train_end = block["train_end"]

        block_train_df = full_df.loc[
            (full_df["event_date"] >= block_start) & (full_df["event_date"] <= train_end)
        ].copy()

        block_train_daily_mean = block_train_df.groupby("event_date")[TARGET_COL].mean().sort_index()
        rs_block = GlobalRollingSeasonalPoissonModel(window_size=WINDOW_SIZE, min_periods=1).fit(
            block_train_daily_mean, daily_mean_full
        )
        base_train = rs_block.predict_for_dates(block_train_df["event_date"]).to_numpy(dtype=float)

        states_train = gather_states_for(block_train_df)

        train_uids = block_train_df["user_id"].to_numpy()
        unique_train_uids, train_user_idx = np.unique(train_uids, return_inverse=True)
        n_users = int(len(unique_train_uids))
        y_train = block_train_df[TARGET_COL].to_numpy(dtype=float)

        lam_u_fit, alpha_fit, _ = fit_joint(
            user_idx=train_user_idx,
            y=y_train,
            b=base_train,
            states=states_train.astype(float),
            n_users=n_users,
            n_alpha=n_alpha,
            lambda_l2=1.0,
            alpha_l2=1e-4,
            max_iter=400,
            verbose=False,
        )
        uid_to_idx = {int(u): i for i, u in enumerate(unique_train_uids)}

        fits[block["block_idx"]] = {
            "lam_u_fit": lam_u_fit,
            "alpha_fit": alpha_fit,
            "uid_to_idx": uid_to_idx,
            "rs_block": rs_block,
            "alpha_norm": float(np.linalg.norm(alpha_fit)),
            "lam_mean": float(lam_u_fit.mean()),
        }
        print(f"  B{block['block_idx'] + 1}: fitted ||α||={fits[block['block_idx']]['alpha_norm']:.4f}, mean λ={fits[block['block_idx']]['lam_mean']:.3f}")

    # === Step 2: for each (k → m) pair, evaluate B_m's test using B_k's joint ===
    print(f"\n=== Evaluating B_k's joint on B_m's test (cross-fold matrix) ===")
    rows: list[dict] = []
    for block_test in blocks:
        m = block_test["block_idx"]
        test_start = block_test["test_start"]
        test_end = block_test["test_end"]
        block_test_df = full_df.loc[
            (full_df["event_date"] >= test_start) & (full_df["event_date"] <= test_end)
        ].copy()
        states_test = gather_states_for(block_test_df)
        test_uids = block_test_df["user_id"].to_numpy()
        y_test = block_test_df[TARGET_COL].to_numpy(dtype=float)

        for block_train in blocks:
            k = block_train["block_idx"]

            # Use B_k's RS to compute baseline on B_m's test dates
            base_test = fits[k]["rs_block"].predict_for_dates(block_test_df["event_date"]).to_numpy(dtype=float)

            # Map test users via B_k's uid_to_idx (default λ=1 for unseen)
            uid_to_idx_k = fits[k]["uid_to_idx"]
            test_user_idx = np.array([uid_to_idx_k.get(int(u), -1) for u in test_uids], dtype=np.int64)
            lam_u_for_test = np.where(
                test_user_idx >= 0,
                fits[k]["lam_u_fit"][np.maximum(test_user_idx, 0)],
                1.0,
            )

            lam_test = lam_u_for_test * base_test + states_test.astype(float) @ fits[k]["alpha_fit"]
            nll = standard_poisson_nll(y_test, lam_test)

            rows.append(
                {
                    "train_block": k,
                    "test_block": m,
                    "k_minus_m": k - m,
                    "nll": nll,
                    "in_fold": (k == m),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "out_of_fold_matrix.csv", index=False)

    # === Aggregates ===
    in_fold = df[df["in_fold"]]
    out_of_fold = df[~df["in_fold"]]
    next_block = df[(df["test_block"] - df["train_block"]) == 1]  # B_k → B_{k+1}

    summary = {
        "n_pairs": int(len(df)),
        "n_blocks": int(len(blocks)),
        "in_fold": {
            "n": int(len(in_fold)),
            "mean_nll": float(in_fold["nll"].mean()),
            "median_nll": float(in_fold["nll"].median()),
            "std_nll": float(in_fold["nll"].std(ddof=1)),
        },
        "out_of_fold_all": {
            "n": int(len(out_of_fold)),
            "mean_nll": float(out_of_fold["nll"].mean()),
            "median_nll": float(out_of_fold["nll"].median()),
            "std_nll": float(out_of_fold["nll"].std(ddof=1)),
        },
        "next_block_only": {
            "n": int(len(next_block)),
            "mean_nll": float(next_block["nll"].mean()),
            "median_nll": float(next_block["nll"].median()),
            "std_nll": float(next_block["nll"].std(ddof=1)),
        },
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"  in-fold       (k=m, n={summary['in_fold']['n']:>3}): mean NLL = {summary['in_fold']['mean_nll']:.4f}")
    print(f"  out-of-fold (k≠m, n={summary['out_of_fold_all']['n']:>3}): mean NLL = {summary['out_of_fold_all']['mean_nll']:.4f}")
    print(f"  next block (k→k+1, n={summary['next_block_only']['n']:>3}): mean NLL = {summary['next_block_only']['mean_nll']:.4f}")

    # === Plot 1: full matrix heatmap ===
    pivot = df.pivot(index="train_block", columns="test_block", values="nll").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    im = ax.imshow(pivot, cmap="viridis", aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            color = "white" if pivot[i, j] > pivot.mean() else "black"
            ax.text(j, i, f"{pivot[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks(range(len(blocks)))
    ax.set_yticks(range(len(blocks)))
    ax.set_xticklabels([f"B{i + 1}" for i in range(len(blocks))])
    ax.set_yticklabels([f"B{i + 1}" for i in range(len(blocks))])
    ax.set_xlabel("Test block (NLL evaluated on this block's test)")
    ax.set_ylabel("Train block (joint fit from this block)")
    ax.set_title("Out-of-fold transfer of joint Hawkes (γ=1): NLL on B_m's test using B_k's fit")
    plt.colorbar(im, ax=ax, label="Test NLL per user-day")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "out_of_fold_matrix.png", dpi=150)
    plt.close(fig)

    # === Plot 2: distribution of in-fold vs out-of-fold ===
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.hist(in_fold["nll"], bins=20, alpha=0.7, color="#2E5EAA", label=f"in-fold (k=m, n={len(in_fold)})", edgecolor="white")
    ax.hist(out_of_fold["nll"], bins=20, alpha=0.6, color="#D2691E", label=f"out-of-fold (k≠m, n={len(out_of_fold)})", edgecolor="white")
    ax.axvline(in_fold["nll"].mean(), color="#0B3C5D", linestyle="--", linewidth=1.4, label=f"in-fold mean = {in_fold['nll'].mean():.4f}")
    ax.axvline(out_of_fold["nll"].mean(), color="#A0522D", linestyle="--", linewidth=1.4, label=f"out-of-fold mean = {out_of_fold['nll'].mean():.4f}")
    ax.set_xlabel("Test NLL per user-day")
    ax.set_ylabel("Count")
    ax.set_title("Joint Hawkes: in-fold vs out-of-fold transfer NLL distribution")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "in_vs_out_of_fold_hist.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Done in {time.time() - t0:.1f}s]")

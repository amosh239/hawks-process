"""Re-plot chapter-10 blockwise CV strip plot with the joint-fit Hawkes column added.

Reads:
  diploma/reports/blockwise_cv/cv_results.csv             — original 6 models
  diploma/reports/joint_lambda_alpha/joint_14d_per_block.csv — joint Hawkes (λ_u + α)

Writes:
  diploma/reports/blockwise_cv/cv_strip_plot.png          — overwritten strip plot
  diploma/reports/blockwise_cv/summary_with_joint.json    — updated summary
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CV_DIR = Path("diploma/reports/blockwise_cv")
JOINT_CSV = Path("diploma/reports/joint_lambda_alpha/joint_14d_per_block.csv")
POOLED_CSV = Path("diploma/reports/blockwise_cv/staged_on_raw_14d.csv")


def main() -> None:
    cv_df = pd.read_csv(CV_DIR / "cv_results.csv")
    joint_df = pd.read_csv(JOINT_CSV)
    pooled_df = pd.read_csv(POOLED_CSV)

    merged = cv_df.merge(
        joint_df[["block_idx", "Joint Hawkes (λ_u + α)"]],
        on="block_idx",
        how="left",
    ).merge(
        pooled_df[["block_idx", "Staged-on-raw (c + alpha)"]].rename(
            columns={"Staged-on-raw (c + alpha)": "Pooled Hawkes (c·b_t + α^T s)"}
        ),
        on="block_idx",
        how="left",
    )

    # Order matches chapter 9 ladder
    model_labels = [
        "Global Poisson",
        "Rolling Poisson",
        "Rolling Seasonal",
        "Personalized Gamma-Poisson",
        "Pooled Hawkes (c·b_t + α^T s)",
        "Scaled-baseline Hawkes",
        "Joint Hawkes (λ_u + α)",
        "GBDT (experimental)",
    ]

    summary = {"n_blocks": int(len(merged)), "models": {}}
    for label in model_labels:
        vals = merged[label].dropna().to_numpy(dtype=float)
        summary["models"][label] = {
            "n": int(len(vals)),
            "mean_nll": float(np.mean(vals)),
            "median_nll": float(np.median(vals)),
            "std_nll": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min_nll": float(np.min(vals)),
            "max_nll": float(np.max(vals)),
        }

    with open(CV_DIR / "summary_with_joint.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Strip plot
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    rng = np.random.default_rng(0)
    # Color scheme: probabilistic models in blue, GBDT orange
    colors = ["#2E5EAA"] * 7 + ["#D2691E"]

    for i, label in enumerate(model_labels):
        vals = merged[label].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        x_jitter = i + (rng.random(len(vals)) - 0.5) * 0.32
        ax.scatter(
            x_jitter, vals, color=colors[i], alpha=0.65, s=42, edgecolors="white", linewidths=0.6
        )
        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        ax.hlines(
            mean_val, i - 0.28, i + 0.28, color="#0B3C5D", linewidth=2.4,
            label="Mean" if i == 0 else None, zorder=4
        )
        ax.hlines(
            median_val, i - 0.28, i + 0.28, color="#D2691E", linewidth=1.8, linestyles="--",
            label="Median" if i == 0 else None, zorder=4
        )
        ax.text(i, mean_val, f"{mean_val:.4f}", ha="center", va="bottom", fontsize=9, color="#0B3C5D", fontweight="bold")

    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels([lbl.replace(" ", "\n", 1) for lbl in model_labels], fontsize=8)
    ax.set_ylabel("Test NLL per user-day (lower is better)")
    ax.set_title(
        f"Blockwise 3-week CV: per-block NLL across {len(merged)} blocks (14d train / 7d test)"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(CV_DIR / "cv_strip_plot.png", dpi=150)
    plt.close(fig)

    # Save merged CSV too
    merged.to_csv(CV_DIR / "cv_results_with_joint.csv", index=False)

    print("Updated mean NLL per model:")
    for label in model_labels:
        m = summary["models"][label]
        print(f"  {label:<32s}  mean={m['mean_nll']:.4f}  median={m['median_nll']:.4f}  std={m['std_nll']:.4f}  (n={m['n']})")


if __name__ == "__main__":
    main()

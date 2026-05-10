"""Re-render the per-user Δ LL plot as a two-panel figure.

Top panel: scatter of per-user Δ LL by test-purchase bucket, clipped to |Δ| ≤ 5.
Bottom panel: per-bucket share of users and share of total purchases.

Used to regenerate plots for chapter 6 and chapter 7.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BUCKETS = ["0", "1", "2", "3-5", "6-10", "11+"]


def to_bucket(p: float) -> str:
    if p == 0: return "0"
    if p == 1: return "1"
    if p == 2: return "2"
    if p <= 5: return "3-5"
    if p <= 10: return "6-10"
    return "11+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--new-col", required=True, help="column with new model per-user LL")
    ap.add_argument("--baseline-col", required=True, help="column with baseline per-user LL")
    ap.add_argument("--purchases-col", default="test_purchases")
    ap.add_argument("--output", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--ylabel-top", default="Δ user-LL (clipped to |Δ| ≤ 5)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.copy()
    df["delta_ll"] = df[args.new_col] - df[args.baseline_col]
    df["bucket"] = df[args.purchases_col].map(to_bucket)
    df["delta_clipped"] = df["delta_ll"].clip(lower=-5.0, upper=5.0)

    bucket_summary = df.groupby("bucket").agg(
        n_users=("delta_ll", "size"),
        mean_delta=("delta_ll", "mean"),
        mean_delta_clipped=("delta_clipped", "mean"),
        purchases_sum=(args.purchases_col, "sum"),
    ).reindex(BUCKETS).fillna(0)

    total_users = int(bucket_summary["n_users"].sum())
    total_purchases = float(bucket_summary["purchases_sum"].sum())
    bucket_summary["pct_users"] = bucket_summary["n_users"] / max(total_users, 1) * 100
    bucket_summary["pct_purchases"] = bucket_summary["purchases_sum"] / max(total_purchases, 1) * 100

    n_clipped = int((df["delta_ll"].abs() > 5.0).sum())

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10.0, 7.4), gridspec_kw={"height_ratios": [3, 1.4]}, sharex=True,
    )

    rng = np.random.default_rng(0)
    bucket_x = {b: i for i, b in enumerate(BUCKETS)}
    for b in BUCKETS:
        sub = df[df["bucket"] == b]["delta_clipped"].to_numpy()
        if len(sub) == 0:
            continue
        x_jitter = bucket_x[b] + (rng.random(len(sub)) - 0.5) * 0.32
        ax_top.scatter(x_jitter, sub, s=8, alpha=0.4, color="#2E5EAA", edgecolors="none")
    for i, b in enumerate(BUCKETS):
        m_val = bucket_summary.loc[b, "mean_delta"]
        n = int(bucket_summary.loc[b, "n_users"])
        if n > 0:
            ax_top.hlines(m_val, i - 0.32, i + 0.32, color="#D2691E", linewidth=2.4, zorder=4)
            ax_top.text(i, m_val, f"{m_val:+.3f}", ha="center", va="bottom",
                        fontsize=9, color="#D2691E", fontweight="bold")
    ax_top.axhline(0, color="#888888", linewidth=0.8)
    ax_top.set_ylabel(args.ylabel_top)
    ax_top.set_title(f"{args.title}\n(всего {total_users:,} юзеров; {n_clipped:,} точек обрезаны по |Δ| > 5)")
    ax_top.grid(axis="y", linestyle=":", alpha=0.5)
    ax_top.spines["top"].set_visible(False); ax_top.spines["right"].set_visible(False)
    ax_top.set_ylim(-5.5, 5.5)

    xs = np.arange(len(BUCKETS))
    ax_bot.plot(xs, bucket_summary["pct_users"].to_numpy(), marker="o",
                color="#2E5EAA", linewidth=1.8, label="% юзеров от всех юзеров")
    ax_bot.plot(xs, bucket_summary["pct_purchases"].to_numpy(), marker="s",
                color="#D2691E", linewidth=1.8, label="% покупок от всех покупок")
    for i, b in enumerate(BUCKETS):
        ax_bot.text(i, bucket_summary.loc[b, "pct_users"],
                    f" {bucket_summary.loc[b, 'pct_users']:.1f}%",
                    fontsize=8, color="#2E5EAA", va="bottom")
        ax_bot.text(i, bucket_summary.loc[b, "pct_purchases"],
                    f" {bucket_summary.loc[b, 'pct_purchases']:.1f}%",
                    fontsize=8, color="#D2691E", va="bottom")
    ax_bot.set_xticks(xs); ax_bot.set_xticklabels(BUCKETS)
    ax_bot.set_xlabel("test purchase count")
    ax_bot.set_ylabel("% от total")
    ax_bot.grid(axis="y", linestyle=":", alpha=0.5)
    ax_bot.spines["top"].set_visible(False); ax_bot.spines["right"].set_visible(False)
    ax_bot.legend(frameon=False, loc="best", fontsize=9)

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")
    print(f"  total users={total_users:,}, total purchases={total_purchases:,.0f}")
    print(f"  clipped points: {n_clipped:,}")
    print("\n  Bucket summary:")
    print(bucket_summary[["n_users", "mean_delta", "mean_delta_clipped", "pct_users", "pct_purchases"]].round(3).to_string())


if __name__ == "__main__":
    main()

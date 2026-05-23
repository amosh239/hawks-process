"""Replot per-channel `baseline_vs_hawkes_nll.png` from existing summary.json.

Tiny helper — does not refit anything, just reads
`diploma/reports/15_cross_channel_hawkes/main_3ch/summary.json`
and renders a 1×3 bar chart (Pers GP vs Scaled Hawkes test NLL per channel).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

OUT = Path("diploma/reports/15_cross_channel_hawkes/main_3ch")
GBDT_DIR = Path("diploma/reports/25_cross_channel_gbdt")
s = json.loads((OUT / "summary.json").read_text())
recs = s["per_channel_metrics"]

gbdt = {}
gbdt_path = GBDT_DIR / "summary.json"
if gbdt_path.exists():
    gbdt = {r["target"]: float(r["test_nll"]) for r in json.loads(gbdt_path.read_text())["per_channel_metrics"]}

ch_colors = {"searches": "#2E5EAA", "to_cart": "#7B3FAA", "to_ord": "#D2691E"}

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
for ax, rec in zip(axes, recs):
    target = rec["target"]
    base = float(rec["baseline_test_nll"])
    hawk = float(rec["hawkes_test_nll"])
    boost = gbdt.get(target)
    color = ch_colors[target]

    labels = ["Pers. GP", "Scaled\nHawkes"]
    vals = [base, hawk]
    colors = ["#888888", color]
    if boost is not None:
        labels.append("GBDT\n(141 фича)")
        vals.append(boost)
        colors.append("#D2691E")

    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.62)
    for rect, val in zip(bars, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            val,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#0B3C5D",
        )
    delta = hawk - base
    pct = delta / base * 100.0
    ax.text(
        0.5,
        0.97,
        f"Δ Hawkes vs GP = {delta:+.4f}  ({pct:+.2f}%)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#D2691E",
        fontweight="bold",
    )
    v_min, v_max = min(vals), max(vals)
    pad = (v_max - v_min) * 0.4 if v_max > v_min else v_max * 0.05
    ax.set_ylim(v_min - pad * 0.5, v_max + pad * 1.5)
    ax.set_ylabel("test NLL (lower is better)", fontsize=9)
    ax.set_title(f"target = {target}", color=color, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

fig.suptitle(
    "Cross-channel test NLL: Personalized GP vs Scaled Hawkes vs GBDT (141 фича, полный feature-engineering), `207d` train",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "baseline_vs_hawkes_nll.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT / 'baseline_vs_hawkes_nll.png'}")

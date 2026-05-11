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
s = json.loads((OUT / "summary.json").read_text())
recs = s["per_channel_metrics"]

ch_colors = {"searches": "#2E5EAA", "to_cart": "#7B3FAA", "to_ord": "#D2691E"}

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
for ax, rec in zip(axes, recs):
    target = rec["target"]
    base = float(rec["baseline_test_nll"])
    hawk = float(rec["hawkes_test_nll"])
    color = ch_colors[target]
    bars = ax.bar(
        ["Pers. GP", "Scaled Hawkes"],
        [base, hawk],
        color=["#888888", color],
        edgecolor="white",
        width=0.55,
    )
    for rect, val in zip(bars, [base, hawk]):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            val,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0B3C5D",
        )
    delta = hawk - base
    pct = delta / base * 100.0  # negative = improvement
    ax.text(
        0.5,
        0.96,
        f"Δ = {delta:+.4f}  ({pct:+.2f}% of baseline)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#D2691E",
        fontweight="bold",
    )
    v_min, v_max = min(base, hawk), max(base, hawk)
    pad = (v_max - v_min) * 0.4 if v_max > v_min else v_max * 0.05
    ax.set_ylim(v_min - pad * 0.5, v_max + pad * 1.5)
    ax.set_ylabel("test NLL (lower is better)", fontsize=9)
    ax.set_title(f"target = {target}", color=color, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Cross-channel Scaled Hawkes vs Personalized Gamma-Poisson на test (`207d` train)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "baseline_vs_hawkes_nll.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT / 'baseline_vs_hawkes_nll.png'}")

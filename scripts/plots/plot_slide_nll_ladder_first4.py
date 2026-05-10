"""Render per-observation NLL ladder for the predefence slides — first 4 models only.

Same logic as run_ladder_summary.py's second chart, but truncated to the four
ladder Poisson baselines (chapters 1-4) and dropping the Hawkes/GBDT bars.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mpl_config = ROOT / ".mplconfig"
mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


SUMMARY = Path("diploma/reports/ladder_summary/ladder_summary.json")
OUT_PATH = Path("presentation/nll_ladder_first4.png")
KEEP_LABELS = [
    "Global Poisson",
    "Rolling Poisson",
    "Rolling Seasonal",
    "Personalized Gamma-Poisson",
]


def main() -> None:
    with open(SUMMARY, "r", encoding="utf-8") as f:
        summary = json.load(f)

    by_label = {m["label"]: m for m in summary["models"]}
    rows = [by_label[lbl] for lbl in KEEP_LABELS]
    nll_values = [r["test_mean_poisson_nll"] for r in rows]
    floor = float(summary["saturated_poisson_nll_floor"])

    deltas = [None] + [nll_values[i] - nll_values[i - 1] for i in range(1, len(nll_values))]

    short = ["Global\nPoisson", "Rolling\nPoisson", "Rolling\nSeasonal", "Personalized\nGamma-Poisson"]
    x = np.arange(len(rows))

    nll_min = floor
    nll_max = max(nll_values)
    span = nll_max - nll_min
    bottom = nll_min - 0.05 * span
    top = nll_max + 0.18 * span

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.bar(
        x,
        [v - bottom for v in nll_values],
        bottom=bottom,
        color="#2E5EAA",
        edgecolor="white",
        width=0.62,
    )
    for i, val in enumerate(nll_values):
        ax.text(i, val, f"{val:.4f}", ha="center", va="bottom",
                fontsize=11, color="#0B3C5D", fontweight="bold")
        if deltas[i] is not None:
            ax.text(i, bottom + 0.015 * span, f"Δ {deltas[i]:+.4f}",
                    ha="center", va="bottom", fontsize=9, color="#D2691E", fontweight="bold")

    ax.plot(x, nll_values, color="#0B3C5D", linewidth=1.4, marker="o", markersize=6, zorder=3)

    ax.axhline(floor, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(len(rows) - 0.5, floor, f"Saturated Poisson floor = {floor:.4f}",
            ha="right", va="bottom", fontsize=10, color="#444444", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=10)
    ax.set_ylim(bottom, top)
    ax.set_ylabel("Test NLL per user-day (lower is better)")
    ax.set_title("Лестница базовых моделей: per-observation NLL на тесте")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

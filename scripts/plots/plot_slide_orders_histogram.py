"""Render a wider-x histogram of daily purchase counts for the predefence slides.

Identical to plot_orders_histogram from src/diploma_baselines/plots.py, but
with x-axis clipped at 12 instead of 6+ — to better illustrate the heavy tail.
"""

from __future__ import annotations

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
import pandas as pd

from src.diploma_baselines.data import filter_date_range, load_daily_grid


TARGET_COL = "to_ord"
ANALYSIS_START = pd.Timestamp("2025-01-15")
ANALYSIS_END = pd.Timestamp("2025-09-30")
CLIP_AT = 12

OUT_PATH = Path("presentation/orders_histogram_slide.png")


def main():
    full_df = load_daily_grid(
        "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv",
        value_cols=[TARGET_COL],
    )
    df = filter_date_range(full_df, start_date=ANALYSIS_START, end_date=ANALYSIS_END)
    values = df[TARGET_COL].to_numpy(dtype=float)
    clipped = np.clip(values, 0, CLIP_AT)
    bins = np.arange(-0.5, CLIP_AT + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.hist(clipped, bins=bins, color="#2E5EAA", edgecolor="white")
    ax.set_xticks(range(0, CLIP_AT + 1))
    labels = [str(i) for i in range(0, CLIP_AT)] + [f"{CLIP_AT}+"]
    ax.set_xticklabels(labels)
    ax.set_xlabel("Daily purchase count")
    ax.set_ylabel("User-days (log scale)")
    ax.set_yscale("log")
    ax.set_title("Распределение числа покупок (user-day) на анализируемом окне")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PATH}")
    print(f"  total user-days = {len(values):,}")
    print(f"  zero user-days  = {(values == 0).sum():,} ({(values == 0).mean() * 100:.2f}%)")
    print(f"  max purchase    = {int(values.max())}")
    print(f"  user-days with > {CLIP_AT}: {(values > CLIP_AT).sum():,}")


if __name__ == "__main__":
    main()

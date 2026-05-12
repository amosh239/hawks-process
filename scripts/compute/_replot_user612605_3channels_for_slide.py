"""Per-channel timeline for user 612605 — for slide 4 ("Данные").

Three panels stacked vertically:
  - top: to_ord
  - middle: to_cart
  - bottom: searches

Vertical bars at days with count > 0 (height = count). Red dashed line
at the chapter-6 train/test split (2025-08-10).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

PANEL_PATH = "data/processed/orbitals/dayuses_cohort_10000_seed42_daily_grid.csv"
OUT = Path("diploma/reports/24_user_lifecycle/user612605_3channels.png")
USER_ID = 612605
START = pd.Timestamp("2025-01-15")
END = pd.Timestamp("2025-09-30")
SPLIT = pd.Timestamp("2025-08-10")

CHANNELS = [
    ("to_ord",   "#D2691E"),
    ("to_cart",  "#7B3FAA"),
    ("searches", "#2E5EAA"),
]

df = pd.read_csv(PANEL_PATH, usecols=["user_id", "event_date", *[c for c, _ in CHANNELS]])
df["event_date"] = pd.to_datetime(df["event_date"])
df = df[(df["user_id"] == USER_ID) & (df["event_date"] >= START) & (df["event_date"] <= END)]
df = df.sort_values("event_date").reset_index(drop=True)

fig, axes = plt.subplots(3, 1, figsize=(11.5, 6.0), sharex=True)
for ax, (ch, color) in zip(axes, CHANNELS):
    counts = df[ch].to_numpy(dtype=float)
    dates = df["event_date"].to_numpy()
    pos = counts > 0
    if pos.any():
        ax.vlines(dates[pos], ymin=0, ymax=counts[pos], color=color, lw=1.4, alpha=0.9)
    ax.axvline(SPLIT, color="#A00", linestyle="--", linewidth=1.0, alpha=0.7)
    mean = counts.mean()
    ax.axhline(mean, color="#888", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(0.005, 0.93, f"mean = {mean:.2f}/day", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="#666")
    ax.set_ylim(0, max(counts.max() * 1.10, 1.0))
    ax.set_ylabel(ch, fontsize=10, fontweight="bold", color=color)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[-1].set_xlabel("date")

fig.suptitle(f"User {USER_ID} — дневные счётчики по 3 каналам воронки", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")

from PIL import Image
print(f"size: {Image.open(OUT).size}")

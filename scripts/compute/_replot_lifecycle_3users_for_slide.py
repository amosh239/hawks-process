"""Render a 3-user slim version of personalized_gp_traces.png for slide 4.

Uses already-saved `user_traces.csv` and `summary.json` from chapter 16, so no
refit. Picks first 3 of the 4 chapter-16 users (active / inactive / Joint wins),
drops the Joint-loses user to fit a 16:9 slide.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("diploma/reports/24_user_lifecycle")
TARGET_COL = "to_ord"
COLOR = "#2E5EAA"

traces = pd.read_csv(OUT_DIR / "user_traces.csv", parse_dates=["event_date"])
s = json.loads((OUT_DIR / "summary.json").read_text())
picked = s["picked_users"][:3]  # active, inactive, joint-wins
split_date = pd.Timestamp(s["split_date"])

fig, axes = plt.subplots(3, 1, figsize=(11.5, 6.2), sharex=True)

for ax, info in zip(axes, picked):
    uid = info["user_id"]
    df = traces[traces["user_id"] == uid].sort_values("event_date")
    dates = df["event_date"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=float)
    lam = df["lam_pers_gp"].to_numpy(dtype=float)

    ax.fill_between(dates, 0, lam, color=COLOR, alpha=0.30, linewidth=0)
    ax.plot(dates, lam, color=COLOR, lw=1.2, alpha=0.95)
    purchases = y > 0
    if purchases.any():
        ax.vlines(dates[purchases], ymin=0, ymax=y[purchases],
                  color="black", lw=1.4, alpha=0.85, zorder=4)
    ax.axvline(split_date, color="#A00", linestyle="--", linewidth=1.0, alpha=0.7)

    y_top = max(float(y.max()) if y.size else 0.0, float(lam.max()) if lam.size else 0.0)
    ax.set_ylim(0, y_top * 1.10 + 1e-3)
    ax.set_ylabel("count / λ̂", fontsize=9)
    ax.set_title(
        f"user {uid} — {info['kind']}: train Y={info['total_y_train']}, test Y={info['total_y_test']}",
        fontsize=10,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[-1].set_xlabel("date")

fig.suptitle("Personalized Gamma-Poisson: дневная интенсивность и фактические покупки", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / "personalized_gp_traces_3users.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR / 'personalized_gp_traces_3users.png'}")

from PIL import Image
im = Image.open(OUT_DIR / "personalized_gp_traces_3users.png")
print(f"size: {im.size}")

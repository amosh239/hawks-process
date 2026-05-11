"""Replot only test_nll_vs_lambda.png for ch19 using saved summary.json.

Avoids rerunning 27 fits when we just want to resize the plot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("diploma/reports/19_joint_reg_sweep")
s = json.loads((OUT / "summary.json").read_text())

LAMBDA_L2_GRID = np.array(s["lambda_l2_grid"])
CHANNELS = s["channels"]
n_ch = len(CHANNELS)
test_nll = np.array(s["test_nll"])  # [n_lambda, n_ch]

pers_test = {row["target"]: row["personalized_test_nll"] for row in s["rows"]}
ch_colors = {"searches": "#2E5EAA", "to_cart": "#7B3FAA", "to_ord": "#D2691E"}

fig, axes = plt.subplots(1, n_ch, figsize=(4.0 * n_ch, 5.0), squeeze=False)
for ci, target in enumerate(CHANNELS):
    ax = axes[0][ci]
    vals = test_nll[:, ci]
    ax.plot(LAMBDA_L2_GRID, vals, marker="o", linewidth=1.8,
            color=ch_colors[target], label="Joint Hawkes")
    i_min = int(np.argmin(vals))
    ax.scatter([LAMBDA_L2_GRID[i_min]], [vals[i_min]],
               s=120, facecolors="none", edgecolors=ch_colors[target],
               linewidths=2.0, zorder=4, label=f"min @ λ={LAMBDA_L2_GRID[i_min]:g}")
    rng = vals.max() - vals.min()
    pad = max(rng * 0.20, 0.0005)
    ax.set_ylim(vals.min() - pad, vals.max() + pad)
    pers_const = pers_test[target]
    ax.text(0.02, 0.97,
            f"Personalized GP baseline:\n  {pers_const:.4f} (gap = {vals.min() - pers_const:+.4f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#666",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(0, max(LAMBDA_L2_GRID) * 1.1)
    ax.set_xlabel(r"$\lambda_{\ell_2}$")
    ax.set_ylabel(f"test NLL ({target})")
    ax.set_title(f"target = {target}", color=ch_colors[target], fontweight="bold")
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

fig.suptitle(
    r"Test NLL vs Joint Hawkes $\lambda_{\ell_2}$ (207d train) — tight y-axis, baseline as text",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT / "test_nll_vs_lambda.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT / 'test_nll_vs_lambda.png'}")

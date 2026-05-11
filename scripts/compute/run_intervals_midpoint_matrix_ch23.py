"""Chapter 23: 10% gain-over-baseline intervals → midpoint α matrix.

Inputs: chapter 22 (unregularized) profile sweep for each (target, source) and
        chapter 19 summary (Personalized GP test NLL per target).

For each (target, source):
  1. Threshold = min_test_NLL + 0.10 · (PG_NLL − min_test_NLL)
     — i.e. allow at most 10% loss of the gain Hawkes makes over Personalized GP.
  2. Find interval [α_lo, α_hi] = largest contiguous α-range around argmin where
     test_NLL ≤ threshold, with linear interpolation between grid points.
  3. midpoint = (α_lo + α_hi) / 2.

Outputs:
  - intervals_table.csv  — per coefficient: anchor, argmin, threshold, α_lo, α_hi, midpoint, width
  - midpoint_matrix.png  — heatmap of the 3×3 midpoint matrix
  - intervals_grid.png   — 3×3 test NLL profiles with highlighted interval bands
  - summary.json         — full numeric record + spectral radius of midpoint matrix
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/Users/amosh239/repo/mkn/diploma/.mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANNELS = ("searches", "to_cart", "to_ord")
THRESHOLD_FRACTION = 0.10
CH_COLORS = {"searches": "#2E5EAA", "to_cart": "#7B3FAA", "to_ord": "#D2691E"}

CH22 = Path("diploma/reports/22_profile_all_alphas_unreg/summary.json")
CH19 = Path("diploma/reports/19_joint_reg_sweep/summary.json")
OUT_DIR = Path("diploma/reports/23_intervals_midpoint_matrix")


def find_interval(grid: np.ndarray, nll: np.ndarray, threshold: float) -> tuple[float, float]:
    """Largest contiguous interval around argmin where nll ≤ threshold; linear interp at edges."""
    argmin = int(np.argmin(nll))
    # Walk left over points whose NLL ≤ threshold
    left = argmin
    while left > 0 and nll[left - 1] <= threshold:
        left -= 1
    if left == 0 and nll[0] <= threshold:
        α_lo = float(grid[0])
    else:
        # nll[left-1] > threshold, nll[left] <= threshold — interp
        x1, y1 = float(grid[left - 1]), float(nll[left - 1])
        x2, y2 = float(grid[left]), float(nll[left])
        frac = (threshold - y1) / (y2 - y1)
        α_lo = x1 + frac * (x2 - x1)
    # Walk right
    right = argmin
    while right < len(grid) - 1 and nll[right + 1] <= threshold:
        right += 1
    if right == len(grid) - 1 and nll[-1] <= threshold:
        α_hi = float(grid[-1])
    else:
        x1, y1 = float(grid[right]), float(nll[right])
        x2, y2 = float(grid[right + 1]), float(nll[right + 1])
        frac = (threshold - y1) / (y2 - y1)
        α_hi = x1 + frac * (x2 - x1)
    return α_lo, α_hi


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading chapter 22 (unregularized) sweep...")
    s22 = json.loads(CH22.read_text())
    print("Loading chapter 19 (regularization sweep) for Personalized GP NLL...")
    s19 = json.loads(CH19.read_text())

    # PG test NLL is per-target; recorded in each row of ch19 — pick first row per target.
    pg_test_nll = {}
    for row in s19["rows"]:
        t = row["target"]
        if t not in pg_test_nll:
            pg_test_nll[t] = float(row["personalized_test_nll"])
    print(f"  Personalized GP test NLL per target: {pg_test_nll}")

    # Iterate over 9 sweeps
    records = []
    for sw in s22["sweeps"]:
        target = sw["target"]
        source = sw["source"]
        anchor = float(sw["anchor_pin_val"])
        grid = np.array(sw["pin_grid"], dtype=float)
        nll = np.array([row["test_nll_mean"] for row in sw["rows"]], dtype=float)
        pg = pg_test_nll[target]
        min_nll = float(nll.min())
        argmin_alpha = float(grid[int(np.argmin(nll))])
        gain = pg - min_nll
        threshold = min_nll + THRESHOLD_FRACTION * gain
        if gain <= 0:
            # Hawkes does not improve over PG — interval-by-threshold is degenerate;
            # report only the argmin point.
            α_lo = α_hi = argmin_alpha
            interval_status = "no_gain_over_PG"
        else:
            α_lo, α_hi = find_interval(grid, nll, threshold)
            interval_status = "ok"
        mid = 0.5 * (α_lo + α_hi)
        width = α_hi - α_lo
        records.append({
            "target_idx": sw["target_idx"],
            "source_idx": sw["source_idx"],
            "target": target,
            "source": source,
            "anchor_pin_val": anchor,
            "pg_test_nll": pg,
            "min_test_nll": min_nll,
            "argmin_alpha": argmin_alpha,
            "gain_over_pg": gain,
            "threshold": threshold,
            "alpha_lo": α_lo,
            "alpha_hi": α_hi,
            "alpha_mid": mid,
            "width": width,
            "interval_status": interval_status,
            "grid": grid.tolist(),
            "test_nll": nll.tolist(),
        })
        print(
            f"  {target:<10s} ← {source:<10s}  anchor={anchor:.4f}  PG={pg:.5f}  "
            f"argmin={argmin_alpha:.4f} (NLL={min_nll:.5f})  gain={gain:+.5f}  "
            f"thr={threshold:.5f}  [{α_lo:.4f}..{α_hi:.4f}]  mid={mid:.4f}  width={width:.4f}  "
            f"{interval_status}"
        )

    # ----- Build midpoint matrix (3×3) -----
    # Raw matrix (all 9 midpoints)
    M_raw = np.zeros((3, 3))
    for r in records:
        M_raw[r["target_idx"], r["source_idx"]] = r["alpha_mid"]

    # Funnel convention: searches → to_cart → to_ord (indices 0 → 1 → 2).
    # Upper triangle (i < j) = upstream-target ← downstream-source — no plausible
    # reaction. Profiles in ch.22 for these cells are flat (range ≈ 0.001..0.003).
    # Zero them out for the final interaction matrix.
    M = M_raw.copy()
    for i in range(3):
        for j in range(3):
            if j > i:
                M[i, j] = 0.0

    eigvals_raw = np.linalg.eigvals(M_raw)
    spectral_radius_raw = float(np.max(np.abs(eigvals_raw)))
    eigvals = np.linalg.eigvals(M)
    spectral_radius = float(np.max(np.abs(eigvals)))
    print(f"\nRaw midpoint matrix M_raw (rows = target, cols = source):")
    for i, t in enumerate(CHANNELS):
        print(f"  {t:<10s}: " + "  ".join(f"{M_raw[i,j]:>7.4f}" for j in range(3)))
    print(f"  eigenvalues = {[f'{abs(e):.4f}' for e in eigvals_raw]}")
    print(f"  spectral radius ρ(M_raw) = {spectral_radius_raw:.4f}")
    print(f"\nFinal lower-triangular matrix M (upper triangle zeroed):")
    for i, t in enumerate(CHANNELS):
        print(f"  {t:<10s}: " + "  ".join(f"{M[i,j]:>7.4f}" for j in range(3)))
    print(f"  eigenvalues = {[f'{abs(e):.4f}' for e in eigvals]}")
    print(f"  spectral radius ρ(M) = {spectral_radius:.4f}")

    # ----- Plot 1: 3×3 grid of test NLL profiles with highlighted intervals -----
    fig, axes = plt.subplots(3, 3, figsize=(3.4 * 3, 2.5 * 3), squeeze=False)
    for r in records:
        i, j = r["target_idx"], r["source_idx"]
        ax = axes[i][j]
        target = r["target"]
        source = r["source"]
        grid = np.array(r["grid"])
        nll = np.array(r["test_nll"])
        color = CH_COLORS[target]

        ax.plot(grid, nll, marker="o", linewidth=2.0, color=color)
        # min marker
        k_min = int(np.argmin(nll))
        ax.scatter([grid[k_min]], [nll[k_min]], s=120, facecolors="none",
                   edgecolors=color, linewidths=2.2, zorder=4,
                   label=f"argmin @ α={grid[k_min]:.3f}")
        # PG horizontal line
        ax.axhline(r["pg_test_nll"], color="#666", linestyle=":", linewidth=1.0,
                   label=f"PG baseline = {r['pg_test_nll']:.4f}")
        # threshold horizontal line
        ax.axhline(r["threshold"], color="#cc4444", linestyle="--", linewidth=1.0,
                   label=f"threshold = {r['threshold']:.4f}")
        # interval band
        ax.axvspan(r["alpha_lo"], r["alpha_hi"], color=color, alpha=0.15,
                   label=f"interval [{r['alpha_lo']:.3f}, {r['alpha_hi']:.3f}]")
        # midpoint vertical
        ax.axvline(r["alpha_mid"], color=color, linestyle="-", linewidth=1.2,
                   label=f"mid = {r['alpha_mid']:.4f}")

        rng = nll.max() - nll.min()
        pad = max(rng * 0.20, 1e-5)
        # extend y range a bit upward so PG line stays visible if PG > max(nll)
        ymax = max(nll.max(), r["pg_test_nll"]) + pad
        ymin = nll.min() - pad
        ax.set_ylim(ymin, ymax)

        ax.set_title(rf"$\alpha[\,{target} \leftarrow {source}\,]$"
                     f"\nwidth = {r['width']:.4f}",
                     color=color, fontweight="bold", fontsize=11)
        if i == 2:
            ax.set_xlabel(rf"pinned $\alpha[\,{target} \leftarrow {source}\,]$",
                          fontsize=10)
        if j == 0:
            ax.set_ylabel("test NLL / n", fontsize=10)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(
        rf"10%-gain intervals over Personalized GP baseline  "
        rf"(threshold = min$+$0.1$\cdot$(PG$-$min), unregularized fit, $\lambda_{{\ell_2}}=0$)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_DIR / "intervals_grid.png", dpi=140)
    plt.close(fig)

    # ----- Plot 2: heatmap of final (lower-triangular) midpoint matrix -----
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    vmax = float(M.max())
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(vmax, 1e-6), aspect="equal")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"from\n{c}" for c in CHANNELS], fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"to\n{c}" for c in CHANNELS], fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i,j]:.4f}", ha="center", va="center",
                    color=("white" if M[i, j] > 0.5 * vmax else "black"), fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"midpoint $\alpha_{i,j}$")
    ax.set_title(
        rf"Hawkes interaction matrix (10%-gain midpoints, upper triangle zeroed)"
        f"\nρ(M) = {spectral_radius:.4f}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "midpoint_matrix.png", dpi=150)
    plt.close(fig)

    # ----- CSV + JSON -----
    pd.DataFrame([
        {k: r[k] for k in (
            "target", "source", "anchor_pin_val", "pg_test_nll", "min_test_nll",
            "argmin_alpha", "gain_over_pg", "threshold",
            "alpha_lo", "alpha_hi", "alpha_mid", "width", "interval_status",
        )}
        for r in records
    ]).to_csv(OUT_DIR / "intervals_table.csv", index=False)

    summary = {
        "channels": list(CHANNELS),
        "threshold_fraction": THRESHOLD_FRACTION,
        "source_ch22": str(CH22),
        "source_ch19": str(CH19),
        "personalized_gp_test_nll": pg_test_nll,
        "records": records,
        "midpoint_matrix_raw": M_raw.tolist(),
        "spectral_radius_raw": spectral_radius_raw,
        "eigenvalues_raw_abs": [float(abs(e)) for e in eigvals_raw],
        "midpoint_matrix": M.tolist(),
        "spectral_radius": spectral_radius,
        "eigenvalues_abs": [float(abs(e)) for e in eigvals],
        "upper_triangle_zeroed": True,
        "upper_triangle_rationale": (
            "Funnel convention: searches → to_cart → to_ord. Upper triangle "
            "(target above source in funnel) has flat NLL profiles in ch22 "
            "(range ≈ 0.001..0.003 nat/n), i.e. coefficients are not identified "
            "by data — zeroed."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\nDone. Saved artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()

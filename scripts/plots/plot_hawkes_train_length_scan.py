"""Plot Hawkes wake-up curve as a function of train length.

Reads summary.json from blockwise CV runs at multiple train lengths
and the chapter-6 long-train summary, builds a single chart showing how
||alpha||_2, learned base scale c and Δ vs personalized evolve with train length.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mpl_config = ROOT / ".mplconfig"
xdg_cache = ROOT / ".cache"
mpl_config.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


# Pull from summary.json of each blockwise sweep
SWEEP_DIRS = [
    ("blockwise_cv_hawkes_reg",     14, 13),  # chapter 11
    ("blockwise_cv_hawkes_reg_4w",  28,  8),  # chapter 11b — 28d
    ("blockwise_cv_hawkes_reg_6w",  42,  5),
    ("blockwise_cv_hawkes_reg_8w",  56,  4),
    ("blockwise_cv_hawkes_reg_12w", 84,  3),
    ("blockwise_cv_hawkes_reg_16w", 112, 2),
]

# Setting E (no regularization) is the cleanest reference because it removes
# any concern about degeneracy being driven by alpha_l2 / scale_l2.
REFERENCE_SETTING = "E: no regularization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Hawkes wake-up curve vs train length")
    parser.add_argument(
        "--reports-root",
        default="diploma/reports",
        help="Path to reports directory",
    )
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/hawkes_train_length_scan",
        help="Directory for chart and CSV",
    )
    return parser.parse_args()


def load_setting(reports_root: Path, dir_name: str) -> dict:
    with open(reports_root / dir_name / "summary.json", "r", encoding="utf-8") as f:
        s = json.load(f)
    for entry in s["settings"]:
        if entry["label"] == REFERENCE_SETTING:
            return entry
    raise KeyError(f"Setting {REFERENCE_SETTING} not found in {dir_name}")


def main() -> None:
    args = parse_args()
    reports_root = Path(args.reports_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for dir_name, train_days, n_blocks_expected in SWEEP_DIRS:
        entry = load_setting(reports_root, dir_name)
        rows.append(
            {
                "train_days": int(train_days),
                "train_weeks": float(train_days / 7),
                "n_blocks": int(entry["n"]),
                "mean_nll": float(entry["mean_nll"]),
                "mean_delta_vs_personalized": float(entry["mean_delta"]),
                "mean_c": float(entry["mean_c"]),
                "mean_alpha_norm": float(entry["mean_alpha_norm"]),
                "degenerate_share": float(entry["degenerate_share"]),
            }
        )

    # Append chapter-6 long-train (207d) reference, computed from chapter-6 summary
    with open(reports_root / "experimental_1_hawkes" / "summary.json", "r", encoding="utf-8") as f:
        ch6 = json.load(f)
    base_nll = float(ch6["test_metrics_personalized_poisson"]["mean_poisson_nll"])
    hawkes_nll = float(ch6["test_metrics_hawkes"]["mean_poisson_nll"])
    long_train_alpha = np.array(ch6["alpha_matrix"][feat] for feat in ch6["alpha_matrix"]) if False else None
    # Compute alpha norm from alpha_matrix dict
    alpha_values = []
    for vals in ch6["alpha_matrix"].values():
        alpha_values.extend(vals)
    alpha_norm = float(np.linalg.norm(np.asarray(alpha_values, dtype=float)))
    rows.append(
        {
            "train_days": 207,
            "train_weeks": 207 / 7,
            "n_blocks": 1,
            "mean_nll": hawkes_nll,
            "mean_delta_vs_personalized": float(hawkes_nll - base_nll),
            "mean_c": float(ch6["learned_base_scale"]),
            "mean_alpha_norm": alpha_norm,
            "degenerate_share": 0.0,
        }
    )

    rows.sort(key=lambda r: r["train_days"])

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "wake_up_curve.csv", index=False)
    print(df.to_string(index=False))

    # 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.5))

    train_days = df["train_days"].to_numpy(dtype=float)

    ax = axes[0]
    ax.plot(train_days, df["mean_alpha_norm"], marker="o", color="#0B3C5D", linewidth=1.6)
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Train length, days")
    ax.set_ylabel("Mean ||α||_2")
    ax.set_title("Hawkes coefficient norm")
    ax.grid(axis="both", linestyle=":", alpha=0.5)
    for x, y, n in zip(train_days, df["mean_alpha_norm"], df["n_blocks"]):
        ax.text(x, y, f"  {y:.4f}", fontsize=8, color="#0B3C5D", va="center")

    ax = axes[1]
    ax.plot(train_days, df["mean_c"], marker="s", color="#D2691E", linewidth=1.6)
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=0.8, label="c = 1 (degenerate)")
    ax.set_xlabel("Train length, days")
    ax.set_ylabel("Mean fitted base scale c")
    ax.set_title("Learned base scale c")
    ax.grid(axis="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    for x, y, n in zip(train_days, df["mean_c"], df["n_blocks"]):
        ax.text(x, y, f"  {y:.3f}", fontsize=8, color="#A0522D", va="center")

    ax = axes[2]
    ax.plot(train_days, df["mean_delta_vs_personalized"], marker="^", color="#2E8B57", linewidth=1.6)
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, label="No improvement vs personalized")
    ax.set_xlabel("Train length, days")
    ax.set_ylabel("Mean Δ test NLL/n vs personalized")
    ax.set_title("Test NLL improvement (lower is better)")
    ax.grid(axis="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    for x, y in zip(train_days, df["mean_delta_vs_personalized"]):
        ax.text(x, y, f"  {y:+.4f}", fontsize=8, color="#2E8B57", va="center")

    fig.suptitle(
        "Hawkes wake-up curve: how train-window length controls degeneracy "
        "(setting E: no regularization)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "wake_up_curve.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved: {output_dir}/wake_up_curve.png")


if __name__ == "__main__":
    main()

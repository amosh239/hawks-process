"""Build chapter-12 plots: ladder graph (with Pooled Hawkes inserted) and
blockwise CV strip plot (with Pooled Hawkes added).

Reads existing artifacts and produces:
  diploma/reports/12_pooled_hawkes/test_loglik_ladder.png
  diploma/reports/12_pooled_hawkes/test_nll_per_obs_ladder.png
  diploma/reports/12_pooled_hawkes/cv_strip_plot.png
  diploma/reports/12_pooled_hawkes/ladder_summary.json
  diploma/reports/12_pooled_hawkes/cv_summary.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mpl_config = ROOT / ".mplconfig"
mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("diploma/reports/12_pooled_hawkes")


# Ladder for chapter 12 = chapter 8 ladder with Pooled Hawkes inserted between
# Personalized Gamma-Poisson and Scaled-baseline Hawkes (NLL-natural position).
LADDER = [
    {
        "label": "Global Poisson",
        "summary_path": "diploma/reports/poisson_baseline/summary.json",
        "metric_key": ["test_metrics", "poisson_loglik"],
        "deviance_key": ["test_metrics", "mean_poisson_deviance"],
        "rows_key": ["test_panel", "rows"],
        "kind": "ladder",
    },
    {
        "label": "Rolling Poisson",
        "summary_path": "diploma/reports/rolling_poisson_baseline/summary.json",
        "metric_key": ["test_metrics_rolling_poisson", "poisson_loglik"],
        "deviance_key": ["test_metrics_rolling_poisson", "mean_poisson_deviance"],
        "rows_key": ["test_panel", "rows"],
        "kind": "ladder",
    },
    {
        "label": "Rolling Seasonal",
        "summary_path": "diploma/reports/rolling_seasonal_poisson_baseline/summary.json",
        "metric_key": ["test_metrics_rolling_seasonal_poisson", "poisson_loglik"],
        "deviance_key": ["test_metrics_rolling_seasonal_poisson", "mean_poisson_deviance"],
        "rows_key": ["test_panel", "rows"],
        "kind": "ladder",
    },
    {
        "label": "Personalized\nGamma-Poisson",
        "summary_path": "diploma/reports/personalized_rolling_seasonal_poisson_baseline/summary.json",
        "metric_key": ["test_metrics_posterior_personalized", "poisson_loglik"],
        "deviance_key": ["test_metrics_posterior_personalized", "mean_poisson_deviance"],
        "rows_key": ["test_panel", "rows"],
        "kind": "ladder",
    },
    {
        "label": "Pooled Hawkes\n(c·b_t + α^T s)",
        "summary_path": "diploma/reports/pooled_hawkes_ch6/summary.json",
        "metric_key": ["test_metrics_pooled_hawkes", "poisson_loglik"],
        "deviance_key": ["test_metrics_pooled_hawkes", "mean_poisson_deviance"],
        "rows_key": ["test_panel", "rows"],
        "kind": "ladder",
        "highlight": True,
    },
    {
        "label": "Scaled-baseline\nHawkes",
        "summary_path": "diploma/reports/experimental_1_hawkes/summary.json",
        "metric_key": ["test_metrics_hawkes", "poisson_loglik"],
        "deviance_key": ["test_metrics_hawkes", "mean_poisson_deviance"],
        "rows_key": ["test_panel", "rows"],
        "kind": "ladder",
    },
    {
        "label": "Joint Hawkes\n(λ_u + α)",
        "summary_path": "diploma/reports/joint_lambda_alpha_ch6/summary.json",
        "metric_key": ["test_metrics_joint_hawkes", "poisson_loglik"],
        "deviance_key": ["test_metrics_joint_hawkes", "mean_poisson_deviance"],
        "rows_key": ["test_panel", "rows"],
        "kind": "ladder",
    },
    {
        "label": "GBDT\n(experimental)",
        "summary_path": "diploma/reports/experimental_2_gbdt/summary.json",
        "metric_key": ["test_metrics_gbdt", "poisson_loglik"],
        "deviance_key": ["test_metrics_gbdt", "mean_poisson_deviance"],
        "rows_key": ["test_panel", "rows"],
        "kind": "experimental",
    },
]


def get_nested(d, keys):
    out = d
    for k in keys:
        out = out[k]
    return out


def build_ladder_plots():
    rows = []
    saturated_estimates = []
    for entry in LADDER:
        path = ROOT / entry["summary_path"]
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        ll = float(get_nested(summary, entry["metric_key"]))
        deviance = float(get_nested(summary, entry["deviance_key"]))
        n = int(get_nested(summary, entry["rows_key"]))
        ll_sat = ll + n * deviance / 2.0
        saturated_estimates.append(ll_sat)
        rows.append({
            "label": entry["label"],
            "kind": entry["kind"],
            "highlight": entry.get("highlight", False),
            "test_poisson_loglik": ll,
            "test_mean_poisson_nll": -ll / n,
            "test_n": n,
            "test_mean_deviance": deviance,
        })

    saturated_ll = float(np.mean(saturated_estimates))
    n_test = rows[0]["test_n"]
    saturated_nll = -saturated_ll / n_test

    ladder_only = [r for r in rows if r["kind"] == "ladder"]
    ladder_values = [r["test_poisson_loglik"] for r in ladder_only]
    ladder_deltas = [None] + [ladder_values[i] - ladder_values[i - 1] for i in range(1, len(ladder_values))]

    deltas = []
    ladder_idx = 0
    for r in rows:
        if r["kind"] == "ladder":
            deltas.append(ladder_deltas[ladder_idx])
            ladder_idx += 1
        else:
            deltas.append(None)

    labels = [r["label"] for r in rows]
    values = [r["test_poisson_loglik"] for r in rows]

    def color_for(r):
        if r.get("highlight"):
            return "#7B3FAA"  # purple for the new model
        if r["kind"] == "ladder":
            return "#2E5EAA"
        return "#D2691E"

    bar_colors = [color_for(r) for r in rows]

    # === LL plot ===
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    x = np.arange(len(labels))
    y_min = min(values)
    y_max = saturated_ll
    y_span = y_max - y_min
    bottom = y_min - 0.06 * y_span
    top = y_max + 0.05 * y_span

    bars = ax.bar(x, [v - bottom for v in values], bottom=bottom, color=bar_colors,
                  edgecolor="white", width=0.62)
    for i, (rect, val) in enumerate(zip(bars, values)):
        ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:,.0f}",
                ha="center", va="bottom", fontsize=10, color="#0B3C5D", fontweight="bold")
        if deltas[i] is not None:
            ax.text(rect.get_x() + rect.get_width() / 2, bottom + 0.015 * y_span,
                    f"Δ {deltas[i]:+,.0f}", ha="center", va="bottom",
                    fontsize=9, color="#D2691E", fontweight="bold")

    ladder_x = [i for i, r in enumerate(rows) if r["kind"] == "ladder"]
    ax.plot(ladder_x, ladder_values, color="#0B3C5D", linewidth=1.4, marker="o", markersize=6, zorder=3)

    ax.axhline(saturated_ll, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(len(labels) - 0.5, saturated_ll, f"Saturated Poisson ceiling = {saturated_ll:,.0f}",
            ha="right", va="bottom", fontsize=10, color="#444444", fontstyle="italic")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(bottom, top)
    ax.set_ylabel("Test Poisson log-likelihood (higher is better)")
    ax.set_title("Лестница моделей с Pooled Hawkes (фиолетовый): test log-likelihood")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "test_loglik_ladder.png", dpi=150)
    plt.close(fig)

    # === NLL plot ===
    nll_values = [-r["test_poisson_loglik"] / r["test_n"] for r in rows]
    ladder_nll = [v for v, r in zip(nll_values, rows) if r["kind"] == "ladder"]
    ladder_nll_deltas = [None] + [ladder_nll[i] - ladder_nll[i - 1] for i in range(1, len(ladder_nll))]
    nll_deltas = []
    ladder_idx = 0
    for r in rows:
        if r["kind"] == "ladder":
            nll_deltas.append(ladder_nll_deltas[ladder_idx])
            ladder_idx += 1
        else:
            nll_deltas.append(None)

    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    nll_min = saturated_nll
    nll_max = max(nll_values)
    nll_span = nll_max - nll_min
    nll_bottom = nll_min - 0.05 * nll_span
    nll_top = nll_max + 0.18 * nll_span

    bars = ax.bar(x, [v - nll_bottom for v in nll_values], bottom=nll_bottom, color=bar_colors,
                  edgecolor="white", width=0.62)
    for i, (rect, val) in enumerate(zip(bars, nll_values)):
        ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, color="#0B3C5D", fontweight="bold")
        if nll_deltas[i] is not None:
            ax.text(rect.get_x() + rect.get_width() / 2, nll_bottom + 0.015 * nll_span,
                    f"Δ {nll_deltas[i]:+.4f}", ha="center", va="bottom",
                    fontsize=9, color="#D2691E", fontweight="bold")
    ax.plot(ladder_x, ladder_nll, color="#0B3C5D", linewidth=1.4, marker="o", markersize=6, zorder=3)
    ax.axhline(saturated_nll, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(len(labels) - 0.5, saturated_nll, f"Saturated Poisson floor = {saturated_nll:.4f}",
            ha="right", va="bottom", fontsize=10, color="#444444", fontstyle="italic")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(nll_bottom, nll_top)
    ax.set_ylabel("Test NLL per user-day (lower is better)")
    ax.set_title("Лестница моделей с Pooled Hawkes (фиолетовый): per-observation NLL")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "test_nll_per_obs_ladder.png", dpi=150)
    plt.close(fig)

    summary_out = {
        "test_n": n_test,
        "saturated_poisson_ceiling": saturated_ll,
        "saturated_poisson_nll_floor": saturated_nll,
        "models": [
            {
                "label": rows[i]["label"].replace("\n", " "),
                "kind": rows[i]["kind"],
                "test_poisson_loglik": rows[i]["test_poisson_loglik"],
                "test_mean_poisson_nll": rows[i]["test_mean_poisson_nll"],
                "delta_vs_prev_ladder": deltas[i],
                "gap_to_saturated_ceiling": saturated_ll - rows[i]["test_poisson_loglik"],
            }
            for i in range(len(rows))
        ],
    }
    with open(OUT_DIR / "ladder_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print("Ladder rendered.")
    for r in rows:
        print(f"  {r['label'].replace(chr(10), ' '):<35s}  LL={r['test_poisson_loglik']:>12,.0f}  NLL={-r['test_poisson_loglik']/r['test_n']:.4f}")


def build_strip_plot():
    cv_df = pd.read_csv("diploma/reports/blockwise_cv/cv_results.csv")
    joint_df = pd.read_csv("diploma/reports/joint_lambda_alpha/joint_14d_per_block.csv")
    pooled_df = pd.read_csv("diploma/reports/blockwise_cv/staged_on_raw_14d.csv")

    merged = cv_df.merge(
        joint_df[["block_idx", "Joint Hawkes (λ_u + α)"]], on="block_idx", how="left"
    ).merge(
        pooled_df[["block_idx", "Staged-on-raw (c + alpha)"]].rename(
            columns={"Staged-on-raw (c + alpha)": "Pooled Hawkes (c·b_t + α^T s)"}
        ),
        on="block_idx",
        how="left",
    )

    model_labels = [
        "Global Poisson",
        "Rolling Poisson",
        "Rolling Seasonal",
        "Personalized Gamma-Poisson",
        "Pooled Hawkes (c·b_t + α^T s)",
        "Scaled-baseline Hawkes",
        "Joint Hawkes (λ_u + α)",
        "GBDT (experimental)",
    ]
    highlight_label = "Pooled Hawkes (c·b_t + α^T s)"

    summary = {"n_blocks": int(len(merged)), "models": {}}
    for label in model_labels:
        vals = merged[label].dropna().to_numpy(dtype=float)
        summary["models"][label] = {
            "n": int(len(vals)),
            "mean_nll": float(np.mean(vals)),
            "median_nll": float(np.median(vals)),
            "std_nll": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min_nll": float(np.min(vals)),
            "max_nll": float(np.max(vals)),
        }
    with open(OUT_DIR / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    merged.to_csv(OUT_DIR / "cv_results_with_pooled.csv", index=False)

    fig, ax = plt.subplots(figsize=(13.6, 6.6))
    rng = np.random.default_rng(0)

    def color_for(label):
        if label == highlight_label:
            return "#7B3FAA"
        if label == "GBDT (experimental)":
            return "#D2691E"
        return "#2E5EAA"

    for i, label in enumerate(model_labels):
        vals = merged[label].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        x_jitter = i + (rng.random(len(vals)) - 0.5) * 0.32
        ax.scatter(x_jitter, vals, color=color_for(label), alpha=0.65, s=42,
                   edgecolors="white", linewidths=0.6)
        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        ax.hlines(mean_val, i - 0.28, i + 0.28, color="#0B3C5D", linewidth=2.4,
                  label="Mean" if i == 0 else None, zorder=4)
        ax.hlines(median_val, i - 0.28, i + 0.28, color="#D2691E", linewidth=1.8,
                  linestyles="--", label="Median" if i == 0 else None, zorder=4)
        ax.text(i, mean_val, f"{mean_val:.4f}", ha="center", va="bottom",
                fontsize=9, color="#0B3C5D", fontweight="bold")

    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels([lbl.replace(" ", "\n", 1) for lbl in model_labels], fontsize=8)
    ax.set_ylabel("Test NLL per user-day (lower is better)")
    ax.set_title(f"Blockwise 3-week CV с Pooled Hawkes: per-block NLL (14d train / 7d test, {len(merged)} блоков)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cv_strip_plot.png", dpi=150)
    plt.close(fig)

    print("\nStrip plot rendered.")
    for label in model_labels:
        m = summary["models"][label]
        print(f"  {label:<32s}  mean={m['mean_nll']:.4f}  median={m['median_nll']:.4f}  std={m['std_nll']:.4f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_ladder_plots()
    build_strip_plot()


if __name__ == "__main__":
    main()

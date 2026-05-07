from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def get_nested(d: dict, keys: list[str]):
    out = d
    for k in keys:
        out = out[k]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the test-LL ladder chart and a summary table from existing run artifacts"
    )
    parser.add_argument(
        "--output-dir",
        default="diploma/reports/ladder_summary",
        help="Directory for the ladder chart and json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    saturated_estimates: list[float] = []
    for entry in LADDER:
        path = ROOT / entry["summary_path"]
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        ll = float(get_nested(summary, entry["metric_key"]))
        deviance = float(get_nested(summary, entry["deviance_key"]))
        n = int(get_nested(summary, entry["rows_key"]))
        ll_sat = ll + n * deviance / 2.0
        saturated_estimates.append(ll_sat)
        rows.append(
            {
                "label": entry["label"],
                "kind": entry["kind"],
                "test_poisson_loglik": ll,
                "test_mean_poisson_nll": -ll / n,
                "test_n": n,
                "test_mean_deviance": deviance,
            }
        )

    saturated_ll = float(np.mean(saturated_estimates))
    n_test = rows[0]["test_n"]
    saturated_nll = -saturated_ll / n_test

    ladder_only = [r for r in rows if r["kind"] == "ladder"]
    ladder_values = [r["test_poisson_loglik"] for r in ladder_only]
    ladder_deltas = [None] + [ladder_values[i] - ladder_values[i - 1] for i in range(1, len(ladder_values))]

    labels = [row["label"] for row in rows]
    values = [row["test_poisson_loglik"] for row in rows]

    deltas: list[float | None] = []
    ladder_idx = 0
    for r in rows:
        if r["kind"] == "ladder":
            deltas.append(ladder_deltas[ladder_idx])
            ladder_idx += 1
        else:
            deltas.append(None)

    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    x = np.arange(len(labels))

    y_min = min(values)
    y_max = saturated_ll
    y_span = y_max - y_min
    bottom = y_min - 0.06 * y_span
    top = y_max + 0.05 * y_span

    bar_colors = ["#2E5EAA" if r["kind"] == "ladder" else "#D2691E" for r in rows]
    bars = ax.bar(
        x,
        [val - bottom for val in values],
        bottom=bottom,
        color=bar_colors,
        edgecolor="white",
        width=0.62,
    )
    for i, (rect, val) in enumerate(zip(bars, values)):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            val,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#0B3C5D",
            fontweight="bold",
        )
        if deltas[i] is not None:
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                bottom + 0.015 * y_span,
                f"Δ {deltas[i]:+,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#D2691E",
                fontweight="bold",
            )

    ladder_x = [i for i, r in enumerate(rows) if r["kind"] == "ladder"]
    ax.plot(
        ladder_x,
        ladder_values,
        color="#0B3C5D",
        linewidth=1.4,
        marker="o",
        markersize=6,
        zorder=3,
    )

    ax.axhline(saturated_ll, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(
        len(labels) - 0.5,
        saturated_ll,
        f"Saturated Poisson ceiling = {saturated_ll:,.0f}",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#444444",
        fontstyle="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(bottom, top)
    ax.set_ylabel("Test Poisson log-likelihood (higher is better)")
    ax.set_title("Лестница моделей: test log-likelihood по ступеням")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "test_loglik_ladder.png", dpi=150)
    plt.close(fig)

    # Second chart: per-observation NLL (lower is better, comparable across windows)
    nll_values = [-r["test_poisson_loglik"] / r["test_n"] for r in rows]
    nll_deltas: list[float | None] = []
    ladder_nll = [v for v, r in zip(nll_values, rows) if r["kind"] == "ladder"]
    ladder_nll_deltas = [None] + [ladder_nll[i] - ladder_nll[i - 1] for i in range(1, len(ladder_nll))]
    ladder_idx = 0
    for r in rows:
        if r["kind"] == "ladder":
            nll_deltas.append(ladder_nll_deltas[ladder_idx])
            ladder_idx += 1
        else:
            nll_deltas.append(None)

    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    nll_min = saturated_nll
    nll_max = max(nll_values)
    nll_span = nll_max - nll_min
    nll_bottom = nll_min - 0.05 * nll_span
    nll_top = nll_max + 0.18 * nll_span

    bars = ax.bar(
        x,
        [val - nll_bottom for val in nll_values],
        bottom=nll_bottom,
        color=bar_colors,
        edgecolor="white",
        width=0.62,
    )
    for i, (rect, val) in enumerate(zip(bars, nll_values)):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            val,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#0B3C5D",
            fontweight="bold",
        )
        if nll_deltas[i] is not None:
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                nll_bottom + 0.015 * nll_span,
                f"Δ {nll_deltas[i]:+.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#D2691E",
                fontweight="bold",
            )

    ax.plot(
        ladder_x,
        ladder_nll,
        color="#0B3C5D",
        linewidth=1.4,
        marker="o",
        markersize=6,
        zorder=3,
    )

    ax.axhline(saturated_nll, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(
        len(labels) - 0.5,
        saturated_nll,
        f"Saturated Poisson floor = {saturated_nll:.4f}",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#444444",
        fontstyle="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(nll_bottom, nll_top)
    ax.set_ylabel("Test NLL per user-day (lower is better)")
    ax.set_title("Лестница моделей: per-observation NLL на тесте")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "test_nll_per_obs_ladder.png", dpi=150)
    plt.close(fig)

    summary_out = {
        "test_n": n_test,
        "saturated_poisson_ceiling": saturated_ll,
        "saturated_poisson_nll_floor": saturated_nll,
        "saturated_estimates_per_model": [
            {"label": rows[i]["label"].replace("\n", " "), "ll_sat_estimate": saturated_estimates[i]}
            for i in range(len(rows))
        ],
        "models": [
            {
                "label": rows[i]["label"].replace("\n", " "),
                "kind": rows[i]["kind"],
                "test_poisson_loglik": rows[i]["test_poisson_loglik"],
                "test_mean_poisson_nll": rows[i]["test_mean_poisson_nll"],
                "delta_vs_prev_ladder": deltas[i],
                "gap_to_saturated_ceiling": saturated_ll - rows[i]["test_poisson_loglik"],
                "share_of_ceiling_gap_closed_vs_step1": (
                    None
                    if i == 0
                    else (rows[i]["test_poisson_loglik"] - rows[0]["test_poisson_loglik"])
                    / (saturated_ll - rows[0]["test_poisson_loglik"])
                ),
            }
            for i in range(len(rows))
        ],
    }
    with open(output_dir / "ladder_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

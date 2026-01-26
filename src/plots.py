import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt


def plot_ll_gain_hist(df, path):
    plt.figure(figsize=(6, 4))
    df["ll_gain"].hist(bins=30, color="steelblue")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("ll_gain (per event)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_ll_vs_events(df, path):
    plt.figure(figsize=(6, 4))
    plt.scatter(df["n_events"], df["ll_gain"], alpha=0.6, color="darkorange")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("# events")
    plt.ylabel("ll_gain")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_alpha_distribution(df, path):
    alpha_cols = [c for c in df.columns if c.startswith("alpha_type_")]
    if not alpha_cols:
        return
    plt.figure(figsize=(6, 4))
    for col in alpha_cols:
        df[col].hist(bins=20, alpha=0.5, label=col)
    plt.legend()
    plt.xlabel("alpha by type")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

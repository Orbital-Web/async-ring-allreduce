import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from utils.plot_family_comparisons import plot_kw_for_impl, sort_impl_labels


def format_bytes(x, pos):
    """Formats axis ticks into readable sizes (KB, MB, GB)."""
    if x == 0:
        return "0 B"

    # 1 KB = 1024 Bytes logic
    if x >= 1024**3:
        return f"{x / 1024**3:.0f} GB"
    elif x >= 1024**2:
        return f"{x / 1024**2:.0f} MB"
    elif x >= 1024:
        return f"{x / 1024:.0f} KB"
    else:
        return f"{x:.0f} B"


def plot_scurve(input_file: str, output_file: str, title: str) -> None:
    # 1. Setup Theme
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "sans-serif"

    # 2. Load Data
    data = pd.read_csv(input_file)
    data["Bandwidth (GB/s)"] = data["throughput"] * 1e6 / (1024**3)  # Convert to GB/s
    data = data[data["impl"] != "NCCL"]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    ordered = sort_impl_labels(data["impl"].unique().tolist())
    for impl in ordered:
        sub = data[data["impl"] == impl].sort_values("input_bytes")
        if sub.empty:
            continue
        kw = plot_kw_for_impl(impl)
        ax.plot(sub["input_bytes"], sub["Bandwidth (GB/s)"], **kw)

    plt.xscale("log")
    ticks = [
        1024,
        1024 * 8,
        1024 * 8 * 8,
        1024 * 8 * 8 * 8,
        1024 * 8 * 8 * 8 * 8,
        1024 * 8 * 8 * 8 * 8 * 8,
        1024 * 8 * 8 * 8 * 8 * 8 * 8,
        1024 * 8 * 8 * 8 * 8 * 8 * 8 * 8,
        1024 * 8 * 8 * 8 * 8 * 8 * 8 * 8 * 4,
    ]
    ax.set_ylim(bottom=0)

    plt.grid(True, which="minor", ls="--", alpha=0.3)
    plt.grid(True, which="major", ls="-", alpha=0.8)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Input Size", fontsize=12)
    plt.ylabel("Per-GPU Bandwidth (GB/s)", fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    # Keep legend sorted and compact
    paired = [(h, lbl) for h, lbl in zip(handles, labels)]
    lbl_order = sort_impl_labels([lbl for _, lbl in paired])
    order_map = {lbl: i for i, lbl in enumerate(lbl_order)}
    paired.sort(key=lambda t: order_map[t[1]])
    h2 = [x[0] for x in paired]
    l2 = [x[1] for x in paired]
    ax.legend(
        h2,
        l2,
        title="Implementation",
        loc="upper left",
        fontsize=10,
        title_fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot S-Curve from benchmark CSV")
    parser.add_argument("input_file", type=str, help="File containing CSV results")
    parser.add_argument("--output", type=str, default="s_curve.png", help="Output filename")
    parser.add_argument("--title", type=str, default="S-Curve Plot", help="Title of the plot")

    args = parser.parse_args()
    plot_scurve(args.input_file, args.output, args.title)

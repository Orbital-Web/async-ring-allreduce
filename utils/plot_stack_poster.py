#!/usr/bin/env python3
"""
Stacked comm vs compute bars from ALLREDUCE_BENCH_STACK=1 baseline CSVs.

Bars use **avg_latency** as total height: comm (`stack_comm_us`) + compute (`stack_compute_us`) + remainder
(uninstrumented gap). Classic vs Pipelined Ring side-by-side per message size.

Example:
    python utils/plot_stack_poster.py \\
        --csv6 results/6r_baseline_stack.csv \\
        --csv8 results/8r_baseline_stack.csv \\
        --output figures/poster/ring_stack_comm_compute_6r_8r.png
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COLOR_COMM = "#9575CD"  # purple
_COLOR_COMP = "#A5D6A7"  # light green
_COLOR_REM = "#D0D0D0"  # remainder / uninstrumented


def load_benchmark_csv(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = next((i for i, L in enumerate(lines) if L.strip().startswith("impl,")), None)
    if start is None:
        raise ValueError(f"No CSV header in {path}")
    return pd.read_csv(StringIO("\n".join(lines[start:])))


def _human_size(nbytes: int) -> str:
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.0f} GiB"
    if nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.0f} MiB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.0f} KiB"
    return f"{nbytes} B"


def pick_sizes(df: pd.DataFrame, *, max_bars: int) -> list[int]:
    """Prefer log-spaced input_bytes present for BOTH ring impls."""
    c = set(df.loc[df["impl"] == "Classic Ring", "input_bytes"].astype(int))
    p = set(df.loc[df["impl"] == "Pipelined Ring", "input_bytes"].astype(int))
    sizes = sorted(c & p)
    if not sizes:
        return []
    if len(sizes) <= max_bars:
        return sizes
    idx = np.round(np.linspace(0, len(sizes) - 1, max_bars)).astype(int)
    return sorted(set(int(sizes[i]) for i in idx))


def plot_tier_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    nbytes_list: list[int],
    tier_label: str,
) -> None:
    classic = df[df["impl"] == "Classic Ring"].set_index("input_bytes")
    piped = df[df["impl"] == "Pipelined Ring"].set_index("input_bytes")

    x = np.arange(len(nbytes_list))
    width = 0.35

    def stacking(row) -> tuple[float, float, float, float]:
        """comm, compute, remainder, wall — all µs"""
        if row is None or row.empty:
            return 0.0, 0.0, 0.0, 0.0
        w = float(row["avg_latency"])
        c = float(row["stack_comm_us"])
        k = float(row["stack_compute_us"])
        rem = max(0.0, w - c - k)
        return c, k, rem, w

    for side, (name, series) in enumerate(
        [("Classic Ring", classic), ("Pipelined Ring", piped)]
    ):
        offset = -width / 2 if side == 0 else width / 2
        xs = x + offset
        comms, comps, rems = [], [], []
        for nb in nbytes_list:
            try:
                r = series.loc[int(nb)]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                c, k, rem, _ = stacking(r)
            except (KeyError, TypeError):
                c, k, rem = 0.0, 0.0, 0.0
            comms.append(c)
            comps.append(k)
            rems.append(rem)

        ax.bar(xs, comms, width, label=("NCCL bracket" if side == 0 else None), color=_COLOR_COMM)
        ax.bar(
            xs,
            comps,
            width,
            bottom=comms,
            label=("add_kernel bracket" if side == 0 else None),
            color=_COLOR_COMP,
        )
        ax.bar(
            xs,
            rems,
            width,
            bottom=[a + b for a, b in zip(comms, comps)],
            label=("remaining wall time" if side == 0 else None),
            color=_COLOR_REM,
        )

    labels = [_human_size(n) for n in nbytes_list]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Time per iteration (µs)")
    ax.set_title(tier_label, fontweight="bold", fontsize=11)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", fontsize=8)


def run(args: argparse.Namespace) -> int:
    paths: list[tuple[str, Path]] = []
    if args.csv6:
        paths.append(("6 ranks", args.csv6))
    if args.csv8:
        paths.append(("8 ranks", args.csv8))
    if not paths:
        print("error: pass --csv6 and/or --csv8", file=sys.stderr)
        return 1

    nrows = len(paths)
    fig, axes = plt.subplots(nrows, 1, figsize=(max(10.0, 2.2 * args.max_sizes), 4.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (tier_label, csv_path) in zip(axes_flat, paths):
        df = load_benchmark_csv(csv_path)
        req = {"stack_comm_us", "stack_compute_us", "avg_latency", "impl", "input_bytes"}
        missing = req - set(df.columns)
        if missing:
            print(f"error: {csv_path} missing columns {missing}", file=sys.stderr)
            return 1
        nbytes_list = pick_sizes(df, max_bars=args.max_sizes)
        plot_tier_ax(ax, df, nbytes_list, tier_label)

    foot = (
        "Stacked: CUDA-event NCCL bracket + add_kernel bracket; top segment = avg_latency − stacks "
        "(sync/overlap/Host). ALLREDUCE_BENCH_STACK perturbs pipelined overlap."
    )
    fig.suptitle(
        "Ring allreduce: communication vs computation (instrumented\n" + foot,
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Poster stacked bars from *_baseline_stack.csv")
    p.add_argument("--csv6", type=Path, help="e.g. results/6r_baseline_stack.csv")
    p.add_argument("--csv8", type=Path, help="e.g. results/8r_baseline_stack.csv")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("figures/poster/ring_stack_comm_compute.png"),
    )
    p.add_argument(
        "--max-sizes",
        type=int,
        default=7,
        help="Max number of message sizes (log-spaced) per tier.",
    )
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

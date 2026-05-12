#!/usr/bin/env python3
"""
Family-style sweep plots (poster-friendly):

- **6r / 8r family layout**: four subplots (2×2) — Pipelined Ring, Classic Ring, then
  Pipelined/Classic Paard (6r) or HD (8r). All **solid** lines; sweep/knob = color.

Three comparison bundles — no point markers.

Examples:
    python utils/plot_family_comparisons.py --tier both --comparison all \\
        --tuning-dir results/Tuning --output-dir plots/tuning/families

By default only **bandwidth** PNGs are written. Add `--plot both` if you also want latency.
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

_IMPL_ORDER = [
    "Pipelined Paard",
    "Classic Paard",
    "Pipelined Ring",
    "Classic Ring",
    "Pipelined HD",
    "Classic HD",
]

# Sweep line colours (cycled if more curves than entries; no pink)
_SWEEP_PALETTE = [
    "#A5D6A7",  # light green
    "#FFEB3B",  # yellow
    "#9575CD",  # purple
    "#42A5F5",  # light blue
]


def _sweep_color_map(sweeps: list) -> dict:
    return {s: _SWEEP_PALETTE[i % len(_SWEEP_PALETTE)] for i, s in enumerate(sweeps)}


def load_benchmark_csv(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = next((i for i, L in enumerate(lines) if L.strip().startswith("impl,")), None)
    if start is None:
        raise ValueError(f"No CSV header in {path}")
    return pd.read_csv(StringIO("\n".join(lines[start:])))


def add_bandwidth_gib_s(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bandwidth_gib_s"] = out["throughput"] * 1e6 / (1024**3)
    return out


def align_runs(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-impl intersection of input_size across all runs."""
    if not dfs:
        return pd.DataFrame()
    labels = list(dfs.keys())
    impls = set.intersection(*(set(d["impl"].unique()) for d in dfs.values()))
    rows: list[pd.DataFrame] = []
    for impl in sorted(impls):
        size_sets = [
            set(dfs[lb].loc[dfs[lb]["impl"] == impl, "input_size"].astype(int)) for lb in labels
        ]
        if not size_sets or any(len(s) == 0 for s in size_sets):
            continue
        common = set.intersection(*size_sets)
        if not common:
            continue
        for lb in labels:
            d = dfs[lb].loc[dfs[lb]["impl"] == impl].copy()
            d = d[d["input_size"].isin(common)]
            d["sweep"] = lb
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def format_bytes_axis(x: float, pos: int) -> str:
    if x <= 0:
        return "0"
    if x >= 1024**3:
        return f"{x / 1024**3:.0f} GB"
    if x >= 1024**2:
        return f"{x / 1024**2:.0f} MB"
    if x >= 1024:
        return f"{x / 1024:.0f} KB"
    return f"{x:.0f} B"


def filter_tier_impls(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    if tier == "6r":
        return df[df["impl"].str.contains("Ring|Paard", regex=True)].copy()
    if tier == "8r":
        return df[df["impl"].str.contains("Ring|HD", regex=True)].copy()
    raise ValueError(tier)


def ordered_impls(present: set[str]) -> list[str]:
    return [x for x in _IMPL_ORDER if x in present] + sorted(present - set(_IMPL_ORDER))


def resolve_files(
    tuning_dir: Path, specs: list[tuple[str, str]], *, need_baseline: str
) -> dict[str, Path]:
    """Map sweep label -> path; exit if baseline missing; warn & skip optional missing."""
    out: dict[str, Path] = {}
    for label, rel in specs:
        p = tuning_dir / rel
        if not p.is_file():
            if label == need_baseline:
                print(f"error: required file missing: {p}", file=sys.stderr)
                sys.exit(1)
            print(f"warning: skip sweep {label!r} — file not found: {p}", file=sys.stderr)
            continue
        out[label] = p
    if len(out) < 2:
        print("error: need at least two CSV files for a comparison", file=sys.stderr)
        sys.exit(1)
    return out


# (sweep_label, filename) — baseline must be first for each bundle
COMPUTE_SPECS_6R = [
    ("baseline", "6r_baseline.csv"),
    ("compute 0 ns", "6r_compute0_ns.csv"),
    # ALLREDUCE_COMPUTE_NS is in nanoseconds: 20000 ns == 20 µs __nanosleep in add_kernel
    ("compute 20000 ns (20 µs)", "6r_compute20000_ns.csv"),
]
INTER_SPECS_6R = [
    ("baseline", "6r_baseline.csv"),
    ("inter 0 µs (hw)", "6r_inter_0us_hwonly.csv"),
    ("inter 50 µs", "6r_inter_50us.csv"),
    ("inter 500 µs", "6r_inter_500us.csv"),
]
BATCH_SPECS_6R = [
    ("baseline", "6r_baseline.csv"),
    ("B = 4", "6r_B4.csv"),
    ("B = 8", "6r_B8.csv"),
]

COMPUTE_SPECS_8R = [
    ("baseline", "8r_baseline.csv"),
    ("compute 0 ns", "8r_compute0_ns.csv"),
    ("compute 20000 ns (20 µs)", "8r_compute20000_ns.csv"),
]
INTER_SPECS_8R = [
    ("baseline", "8r_baseline.csv"),
    ("inter 0 µs (hw)", "8r_inter_0us_hwonly.csv"),
    ("inter 50 µs", "8r_inter_50us.csv"),
    ("inter 500 µs", "8r_inter_500us.csv"),
]
BATCH_SPECS_8R = [
    ("baseline", "8r_baseline.csv"),
    ("B = 4", "8r_B4.csv"),
    ("B = 8", "8r_B8.csv"),
]


def plot_suptitle(tier: str, comparison: str, *, latency: bool) -> str:
    """Short single-line poster title (no layout / subplot notes)."""
    n = 8 if tier == "8r" else 6
    other = "HD" if tier == "8r" else "Paard"
    if comparison == "batch":
        line = f"{n}-rank varying micro-batch B: Ring vs {other} variants"
    elif comparison == "compute":
        line = f"{n}-rank varying add_kernel compute cost: Ring vs {other} variants"
    elif comparison == "inter":
        line = f"{n}-rank varying inter-node delay: Ring vs {other} variants"
    else:
        raise ValueError(comparison)
    return f"{line} (latency)" if latency else line


def plot_faceted_sweep(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    log_y: bool,
    out_path: Path,
    title: str,
) -> None:
    present = set(df["impl"].unique())
    impl_list = ordered_impls(present)
    n = len(impl_list)
    ncol = 2
    nrow = (n + ncol - 1) // ncol

    sweeps = sorted(pd.unique(df["sweep"]), key=lambda s: (s != "baseline", s))
    sweep_color = _sweep_color_map(sweeps)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 4.2 * nrow), squeeze=False)
    axes_f = axes.ravel()

    for ax, impl in zip(axes_f, impl_list):
        sub = df[df["impl"] == impl].sort_values("input_bytes")
        for sw in sweeps:
            ss = sub[sub["sweep"] == sw]
            if ss.empty:
                continue
            ax.plot(
                ss["input_bytes"],
                ss[metric],
                label=sw,
                color=sweep_color[sw],
                linestyle="-",
                linewidth=2.5,
                marker=None,
            )
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes_axis))
        ax.set_title(impl, fontweight="bold", fontsize=10)
        ax.set_xlabel("Input size (bytes)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="minor", ls="--", alpha=0.3)
        ax.legend(fontsize=8, title="Sweep", title_fontsize=9)

    for j in range(len(impl_list), len(axes_f)):
        axes_f[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_family_four_panel_sweep(
    df: pd.DataFrame,
    *,
    tier: str,
    metric: str,
    ylabel: str,
    log_y: bool,
    out_path: Path,
    title: str,
) -> None:
    """
    Four panels (2×2), same style as per-impl faceted plots:
    top row = Ring (Pipelined | Classic); bottom row = Paard (6r) or HD (8r).
    Colors = sweep; all solid lines.
    """
    if tier == "8r":
        grid = [
            ["Pipelined Ring", "Classic Ring"],
            ["Pipelined HD", "Classic HD"],
        ]
    elif tier == "6r":
        grid = [
            ["Pipelined Ring", "Classic Ring"],
            ["Pipelined Paard", "Classic Paard"],
        ]
    else:
        raise ValueError(tier)

    present = set(df["impl"].unique())
    sweeps = sorted(pd.unique(df["sweep"]), key=lambda s: (s != "baseline", s))
    sweep_color = _sweep_color_map(sweeps)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), squeeze=False)

    for i in range(2):
        for j in range(2):
            impl = grid[i][j]
            ax = axes[i, j]
            if impl not in present:
                ax.set_visible(False)
                continue
            sub = df[df["impl"] == impl].sort_values("input_bytes")
            for sw in sweeps:
                ss = sub[sub["sweep"] == sw]
                if ss.empty:
                    continue
                ax.plot(
                    ss["input_bytes"],
                    ss[metric],
                    label=sw,
                    color=sweep_color[sw],
                    linestyle="-",
                    linewidth=2.5,
                    marker=None,
                )
            ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes_axis))
            ax.set_title(impl, fontweight="bold", fontsize=10)
            ax.set_xlabel("Input size (bytes)")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="minor", ls="--", alpha=0.3)
            ax.legend(fontsize=8, title="Sweep", title_fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_comparison(
    *,
    tier: str,
    comparison: str,
    tuning_dir: Path,
    output_dir: Path,
    plot: str,
    layout: str,
) -> None:
    if comparison == "compute":
        specs = COMPUTE_SPECS_6R if tier == "6r" else COMPUTE_SPECS_8R
        tag = "compute_sweep"
    elif comparison == "inter":
        specs = INTER_SPECS_6R if tier == "6r" else INTER_SPECS_8R
        tag = "inter_sweep"
    elif comparison == "batch":
        specs = BATCH_SPECS_6R if tier == "6r" else BATCH_SPECS_8R
        tag = "batch_sweep"
    else:
        raise ValueError(comparison)

    paths = resolve_files(tuning_dir, specs, need_baseline="baseline")
    dfs = {lab: load_benchmark_csv(p) for lab, p in paths.items()}
    merged = align_runs(dfs)
    merged = filter_tier_impls(merged, tier)
    merged = add_bandwidth_gib_s(merged)

    if merged.empty:
        print(f"warning: no overlapping data for {tier} {comparison}", file=sys.stderr)
        return

    if layout == "family":
        sub = f"{tier}_{tag}_by_family"
    else:
        sub = f"{tier}_{tag}"
    base = output_dir / sub

    title_bw = plot_suptitle(tier, comparison, latency=False)
    title_lat = plot_suptitle(tier, comparison, latency=True)
    if plot in ("bandwidth", "both"):
        if layout == "family":
            plot_family_four_panel_sweep(
                merged,
                tier=tier,
                metric="bandwidth_gib_s",
                ylabel="Per-GPU bandwidth (GiB/s)",
                log_y=False,
                out_path=Path(f"{base}_bandwidth.png"),
                title=title_bw,
            )
        else:
            plot_faceted_sweep(
                merged,
                metric="bandwidth_gib_s",
                ylabel="Per-GPU bandwidth (GiB/s)",
                log_y=False,
                out_path=Path(f"{base}_bandwidth.png"),
                title=title_bw,
            )
        print(f"wrote {base}_bandwidth.png")
    if plot in ("latency", "both"):
        if layout == "family":
            plot_family_four_panel_sweep(
                merged,
                tier=tier,
                metric="avg_latency",
                ylabel="Average latency (µs)",
                log_y=True,
                out_path=Path(f"{base}_latency.png"),
                title=title_lat,
            )
        else:
            plot_faceted_sweep(
                merged,
                metric="avg_latency",
                ylabel="Average latency (µs)",
                log_y=True,
                out_path=Path(f"{base}_latency.png"),
                title=title_lat,
            )
        print(f"wrote {base}_latency.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ring vs Paard/HD family sweep plots.")
    parser.add_argument("--tier", choices=("6r", "8r", "both"), default="both")
    parser.add_argument(
        "--comparison",
        choices=("compute", "inter", "batch", "all"),
        default="all",
        help="Which bundle of CSVs to compare.",
    )
    parser.add_argument(
        "--tuning-dir",
        type=Path,
        default=Path("results/Tuning"),
        help="Directory containing 6r_*.csv / 8r_*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/tuning/families"),
    )
    parser.add_argument(
        "--plot",
        choices=("bandwidth", "latency", "both"),
        default="bandwidth",
        help="What to plot (default: bandwidth only; throughput S-curves).",
    )
    parser.add_argument(
        "--layout",
        choices=("per-impl", "family"),
        default="family",
        help="'family': 2×2 subplots — Ring row, Paard/HD row; pipelined/classic split; solid lines. "
        "'per-impl': one subplot per algorithm (column order from CSV set).",
    )
    args = parser.parse_args()

    comps = ["compute", "inter", "batch"] if args.comparison == "all" else [args.comparison]
    tiers = ["6r", "8r"] if args.tier == "both" else [args.tier]

    for tier in tiers:
        for c in comps:
            run_comparison(
                tier=tier,
                comparison=c,
                tuning_dir=args.tuning_dir,
                output_dir=args.output_dir,
                plot=args.plot,
                layout=args.layout,
            )

    print(
        "\nNote: 8r inter sweep expects 8r_inter_50us.csv and 8r_inter_500us.csv if you "
        "ran those experiments; missing files are skipped with a warning. "
        "Batch sweeps require B4/B8 CSVs (e.g. 8r_B4.csv) — add runs or symlinks if missing.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

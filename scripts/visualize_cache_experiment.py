#!/usr/bin/env python3
"""Visualize SeaCache refresh/quality tradeoffs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SeaCache cache ratio and image quality metrics.")
    parser.add_argument("--quality-summary", required=True, help="JSON summary from evaluate_pairwise_quality.py.")
    parser.add_argument(
        "--cache-analysis",
        action="append",
        default=[],
        help="Config cache analysis as name=path. Repeat once per config.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to write plots and merged summaries.")
    parser.add_argument("--order", default=None, help="Optional comma-separated config order.")
    parser.add_argument("--title", default="SeaCache FLUX Experiment", help="Plot title prefix.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected name=path, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty config name in: {value}")
    return name, Path(path)


def load_cache_analyses(items: List[str]) -> Dict[str, Dict[str, Any]]:
    analyses = {}
    for item in items:
        name, path = parse_named_path(item)
        analyses[name] = load_json(path)
    return analyses


def merge_results(
    quality_rows: List[Dict[str, Any]],
    cache_analyses: Dict[str, Dict[str, Any]],
    order: Optional[List[str]],
) -> List[Dict[str, Any]]:
    quality_by_name = {row["config"]: row for row in quality_rows}
    names = order or list(quality_by_name.keys())

    rows = []
    for name in names:
        quality = quality_by_name.get(name)
        if quality is None:
            continue
        cache = cache_analyses.get(name, {})
        rows.append(
            {
                "config": name,
                "num_pairs": quality.get("num_pairs"),
                "refresh_ratio": cache.get("refresh_ratio"),
                "refresh_steps": cache.get("num_refresh"),
                "skip_steps": cache.get("num_skip"),
                "total_steps": cache.get("total_steps"),
                "mean_psnr_db": quality.get("mean_psnr_db"),
                "std_psnr_db": quality.get("std_psnr_db"),
                "mean_lpips": quality.get("mean_lpips"),
                "std_lpips": quality.get("std_lpips"),
                "mean_mse": quality.get("mean_mse"),
                "mean_rmse": quality.get("mean_rmse"),
                "mean_mae": quality.get("mean_mae"),
                "missing_pairs": quality.get("missing_pairs"),
            }
        )
    add_deltas(rows)
    return rows


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def add_deltas(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    baseline = next((row for row in rows if row["config"] == "baseline"), rows[0])
    baseline_refresh = as_float(baseline.get("refresh_ratio"))
    baseline_psnr = as_float(baseline.get("mean_psnr_db"))
    baseline_lpips = as_float(baseline.get("mean_lpips"))

    for row in rows:
        refresh = as_float(row.get("refresh_ratio"))
        psnr = as_float(row.get("mean_psnr_db"))
        lpips = as_float(row.get("mean_lpips"))
        row["delta_refresh_ratio_vs_baseline"] = (
            None if refresh is None or baseline_refresh is None else refresh - baseline_refresh
        )
        row["delta_psnr_db_vs_baseline"] = None if psnr is None or baseline_psnr is None else psnr - baseline_psnr
        row["delta_lpips_vs_baseline"] = None if lpips is None or baseline_lpips is None else lpips - baseline_lpips


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "config",
        "refresh_ratio",
        "mean_psnr_db",
        "mean_lpips",
        "delta_refresh_ratio_vs_baseline",
        "delta_psnr_db_vs_baseline",
        "delta_lpips_vs_baseline",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = [format_value(row.get(header)) for header in headers]
            f.write("| " + " | ".join(values) + " |\n")


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plots. Install with: python -m pip install matplotlib") from exc
    return plt


def annotate_bars(ax, bars, fmt="{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_refresh_ratio(plt, rows: List[Dict[str, Any]], out_dir: Path, title: str) -> None:
    rows = [row for row in rows if row.get("refresh_ratio") is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [row["config"] for row in rows]
    values = [row["refresh_ratio"] for row in rows]
    bars = ax.bar(labels, values, color="#4C78A8")
    annotate_bars(ax, bars)
    ax.set_ylabel("Refresh ratio")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title(f"{title}: Compute Cost")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "refresh_ratio.png", dpi=200)
    plt.close(fig)


def plot_quality_bars(plt, rows: List[Dict[str, Any]], out_dir: Path, title: str) -> None:
    labels = [row["config"] for row in rows]
    psnr = [row.get("mean_psnr_db") for row in rows]
    psnr_std = [row.get("std_psnr_db") or 0.0 for row in rows]
    lpips = [row.get("mean_lpips") for row in rows]
    lpips_std = [row.get("std_lpips") or 0.0 for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    psnr_bars = axes[0].bar(labels, psnr, yerr=psnr_std, color="#59A14F", capsize=4)
    annotate_bars(axes[0], psnr_bars)
    axes[0].set_ylabel("PSNR vs full reference (dB, higher better)")
    axes[0].set_title("Reference Fidelity: PSNR")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)

    lpips_bars = axes[1].bar(labels, lpips, yerr=lpips_std, color="#E15759", capsize=4)
    annotate_bars(axes[1], lpips_bars, fmt="{:.4f}")
    axes[1].set_ylabel("LPIPS vs full reference (lower better)")
    axes[1].set_title("Perceptual Distance: LPIPS")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(f"{title}: Image Quality")
    fig.tight_layout()
    fig.savefig(out_dir / "quality_bars.png", dpi=200)
    plt.close(fig)


def plot_tradeoff(plt, rows: List[Dict[str, Any]], out_dir: Path, title: str) -> None:
    rows = [
        row
        for row in rows
        if row.get("refresh_ratio") is not None and row.get("mean_psnr_db") is not None and row.get("mean_lpips") is not None
    ]
    if not rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    refresh = [row["refresh_ratio"] for row in rows]
    psnr = [row["mean_psnr_db"] for row in rows]
    lpips = [row["mean_lpips"] for row in rows]
    labels = [row["config"] for row in rows]

    axes[0].scatter(refresh, psnr, s=80, color="#59A14F")
    for x, y, label in zip(refresh, psnr, labels):
        axes[0].annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    axes[0].set_xlabel("Refresh ratio (lower faster)")
    axes[0].set_ylabel("PSNR (higher better)")
    axes[0].set_title("Quality/Compute: PSNR")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(refresh, lpips, s=80, color="#E15759")
    for x, y, label in zip(refresh, lpips, labels):
        axes[1].annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    axes[1].set_xlabel("Refresh ratio (lower faster)")
    axes[1].set_ylabel("LPIPS (lower better)")
    axes[1].set_title("Quality/Compute: LPIPS")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"{title}: Tradeoff")
    fig.tight_layout()
    fig.savefig(out_dir / "quality_compute_tradeoff.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quality_rows = load_json(Path(args.quality_summary))
    cache_analyses = load_cache_analyses(args.cache_analysis)
    order = [item.strip() for item in args.order.split(",") if item.strip()] if args.order else None
    rows = merge_results(quality_rows, cache_analyses, order)

    with (out_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    write_csv(out_dir / "experiment_summary.csv", rows)
    write_markdown(out_dir / "experiment_summary.md", rows)

    plt = require_matplotlib()
    plot_refresh_ratio(plt, rows, out_dir, args.title)
    plot_quality_bars(plt, rows, out_dir, args.title)
    plot_tradeoff(plt, rows, out_dir, args.title)

    print(f"Wrote summary and plots to: {out_dir}")
    for row in rows:
        print(
            f"{row['config']}: refresh={format_value(row.get('refresh_ratio'))}, "
            f"PSNR={format_value(row.get('mean_psnr_db'))}, "
            f"LPIPS={format_value(row.get('mean_lpips'))}"
        )


if __name__ == "__main__":
    main()

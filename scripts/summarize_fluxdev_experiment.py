#!/usr/bin/env python3
"""Build final formal FLUX-dev SeaCache summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_ROOT = "experiments/fluxdev_150_v1"
BASELINE_CONFIG = "seacache_baseline_acc030"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the formal FLUX-dev SeaCache experiment.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Experiment root directory.")
    parser.add_argument(
        "--quality-csv",
        default=None,
        help="Per-image quality CSV. Defaults to ROOT/summaries/final_pairwise_quality.csv.",
    )
    parser.add_argument(
        "--quality-summary",
        default=None,
        help="Quality summary JSON. Defaults to ROOT/summaries/final_pairwise_quality_summary.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", "None", "n/a"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(nums) / len(nums) if nums else None


def win_rate(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(1 for value in nums if value > 0.0) / len(nums) if nums else None


def reason_count(summary: Dict[str, Any], key: str) -> int:
    if f"{key}_trigger_refresh_count" in summary:
        return int(summary[f"{key}_trigger_refresh_count"])
    return int(summary.get("refresh_reason", {}).get(key, {}).get("count", 0))


def load_config_metadata(root: Path) -> Dict[str, Dict[str, Any]]:
    selected_path = root / "configs" / "selected_dual_configs.json"
    selected = load_json(selected_path)["selected"] if selected_path.exists() else {}
    return {
        "full_compute_reference": {
            "config": "full_compute_reference",
            "cache_gate": "disabled",
            "delta_acc": None,
            "delta_single": None,
            "delta_single_quantile": None,
            "cache_disabled": True,
            "reference_mode": "seacache_disabled",
        },
        "seacache_baseline_acc030": {
            "config": "seacache_baseline_acc030",
            "cache_gate": "accumulated",
            "delta_acc": 0.30,
            "delta_single": None,
            "delta_single_quantile": None,
        },
        "dual_q95_matched": {
            "config": "dual_q95_matched",
            "cache_gate": "dual",
            **selected.get("dual_q95_matched", {}),
        },
        "dual_q925_matched": {
            "config": "dual_q925_matched",
            "cache_gate": "dual",
            **selected.get("dual_q925_matched", {}),
        },
        "dual_q975_matched": {
            "config": "dual_q975_matched",
            "cache_gate": "dual",
            **selected.get("dual_q975_matched", {}),
        },
        "dual_q95_same_acc": {
            "config": "dual_q95_same_acc",
            "cache_gate": "dual",
            **selected.get("dual_q95_same_acc", {}),
        },
    }


def final_order(root: Path) -> List[str]:
    settings_path = root / "configs" / "base_settings.json"
    settings = load_json(settings_path) if settings_path.exists() else {}
    final_settings = settings.get("final_eval", {})
    order = ["full_compute_reference", "seacache_baseline_acc030"]
    if final_settings.get("include_dual_q925_matched", False):
        order.append("dual_q925_matched")
    order.extend(["dual_q95_matched", "dual_q975_matched"])
    if final_settings.get("include_dual_q95_same_acc", True):
        order.append("dual_q95_same_acc")
    return order


def load_run_metadata(root: Path, config: str) -> Dict[str, Any]:
    path = root / "logs" / config / "run_metadata.json"
    if not path.exists():
        return {}
    return load_json(path)


def run_time(root: Path, config: str) -> Tuple[Optional[float], Optional[float]]:
    metadata = load_run_metadata(root, config)
    if not metadata:
        return None, None
    elapsed = as_float(metadata.get("elapsed_seconds"))
    expected_images = as_float(metadata.get("expected_images"))
    if elapsed is None or expected_images in (None, 0.0):
        return elapsed, None
    return elapsed, elapsed / expected_images


def cache_summary_rows(root: Path, metadata: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    settings_path = root / "configs" / "base_settings.json"
    settings = load_json(settings_path) if settings_path.exists() else {}
    num_steps = int(settings.get("num_inference_steps", 0) or 0)
    for config in final_order(root):
        cache_path = root / "summaries" / "final_cache" / f"{config}_cache.json"
        cache = load_json(cache_path) if cache_path.exists() else {}
        total_time, mean_time = run_time(root, config)
        run_metadata = load_run_metadata(root, config)
        expected_images = as_float(run_metadata.get("expected_images"))
        meta = metadata.get(config, {})
        if not cache and meta.get("cache_disabled") and expected_images is not None and num_steps:
            total_steps = int(expected_images) * num_steps
            cache = {
                "refresh_ratio": 1.0,
                "num_refresh": total_steps,
                "num_skip": 0,
                "total_steps": total_steps,
                "refresh_reason": {},
            }
        rows.append(
            {
                "config": config,
                "cache_gate": meta.get("cache_gate"),
                "delta_acc": meta.get("delta_acc"),
                "delta_single": meta.get("delta_single"),
                "delta_single_quantile": meta.get("delta_single_quantile"),
                "refresh_ratio": cache.get("refresh_ratio"),
                "refresh_steps": cache.get("num_refresh"),
                "skip_steps": cache.get("num_skip"),
                "single_trigger_refresh_count": reason_count(cache, "single"),
                "acc_trigger_refresh_count": reason_count(cache, "acc"),
                "warmup_refresh_count": reason_count(cache, "warmup"),
                "total_steps": cache.get("total_steps"),
                "total_inference_time_seconds": total_time,
                "mean_inference_time_seconds": mean_time,
                "cache_summary_path": str(cache_path) if cache_path.exists() else None,
            }
        )
    return rows


def key_for(row: Dict[str, str]) -> Tuple[int, int]:
    return int(row["prompt_id"]), int(row["seed"])


def build_improvement_rows(quality_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_config: Dict[str, Dict[Tuple[int, int], Dict[str, str]]] = defaultdict(dict)
    for row in quality_rows:
        by_config[row["config"]][key_for(row)] = row

    baseline = by_config.get(BASELINE_CONFIG, {})
    improvements: List[Dict[str, Any]] = []
    for config, rows_by_key in sorted(by_config.items()):
        if config == BASELINE_CONFIG:
            continue
        for key, row in sorted(rows_by_key.items()):
            base = baseline.get(key)
            if base is None:
                continue
            base_psnr = as_float(base.get("psnr_db"))
            cfg_psnr = as_float(row.get("psnr_db"))
            base_lpips = as_float(base.get("lpips"))
            cfg_lpips = as_float(row.get("lpips"))
            base_mae = as_float(base.get("mae"))
            cfg_mae = as_float(row.get("mae"))
            prompt_id, seed = key
            improvements.append(
                {
                    "config": config,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "baseline_psnr_db": base_psnr,
                    "config_psnr_db": cfg_psnr,
                    "psnr_improvement_over_baseline": None
                    if base_psnr is None or cfg_psnr is None
                    else cfg_psnr - base_psnr,
                    "baseline_lpips": base_lpips,
                    "config_lpips": cfg_lpips,
                    "lpips_improvement_over_baseline": None
                    if base_lpips is None or cfg_lpips is None
                    else base_lpips - cfg_lpips,
                    "baseline_mae": base_mae,
                    "config_mae": cfg_mae,
                    "mae_improvement_over_baseline": None
                    if base_mae is None or cfg_mae is None
                    else base_mae - cfg_mae,
                    "reference_image": row.get("reference_image"),
                    "candidate_image": row.get("candidate_image"),
                }
            )
    return improvements


def worst_baseline_keys(
    quality_rows: List[Dict[str, str]], metric: str, higher_is_worse: bool
) -> List[Tuple[int, int]]:
    baseline_rows = [row for row in quality_rows if row["config"] == BASELINE_CONFIG]
    valued = [(key_for(row), as_float(row.get(metric))) for row in baseline_rows]
    valued = [(key, value) for key, value in valued if value is not None]
    if not valued:
        return []
    valued.sort(key=lambda item: item[1], reverse=higher_is_worse)
    k = max(1, math.ceil(len(valued) * 0.10))
    return [key for key, _ in valued[:k]]


def add_improvement_summary(
    quality_summary: List[Dict[str, Any]],
    quality_rows: List[Dict[str, str]],
    improvement_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_config: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in improvement_rows:
        by_config[row["config"]].append(row)

    worst_psnr_keys = set(worst_baseline_keys(quality_rows, "psnr_db", higher_is_worse=False))
    worst_lpips_keys = set(worst_baseline_keys(quality_rows, "lpips", higher_is_worse=True))

    out: List[Dict[str, Any]] = []
    for row in quality_summary:
        config = row["config"]
        improvements = by_config.get(config, [])
        row = dict(row)
        row["mean_psnr_improvement_over_baseline"] = mean(
            item.get("psnr_improvement_over_baseline") for item in improvements
        )
        row["mean_lpips_improvement_over_baseline"] = mean(
            item.get("lpips_improvement_over_baseline") for item in improvements
        )
        row["psnr_win_rate_over_baseline"] = win_rate(
            item.get("psnr_improvement_over_baseline") for item in improvements
        )
        row["lpips_win_rate_over_baseline"] = win_rate(
            item.get("lpips_improvement_over_baseline") for item in improvements
        )
        row["worst10_baseline_psnr_mean_improvement"] = mean(
            item.get("psnr_improvement_over_baseline")
            for item in improvements
            if (int(item["prompt_id"]), int(item["seed"])) in worst_psnr_keys
        )
        row["worst10_baseline_lpips_mean_improvement"] = mean(
            item.get("lpips_improvement_over_baseline")
            for item in improvements
            if (int(item["prompt_id"]), int(item["seed"])) in worst_lpips_keys
        )
        out.append(row)
    return out


def final_table_rows(
    cache_rows: List[Dict[str, Any]],
    quality_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    quality_by_config = {row["config"]: row for row in quality_rows}
    rows = []
    for cache in cache_rows:
        quality = quality_by_config.get(cache["config"], {})
        rows.append(
            {
                "config": cache["config"],
                "delta_acc": cache.get("delta_acc"),
                "delta_single": cache.get("delta_single"),
                "refresh_ratio": cache.get("refresh_ratio"),
                "refresh_steps": cache.get("refresh_steps"),
                "skip_steps": cache.get("skip_steps"),
                "single_trigger_refresh_count": cache.get("single_trigger_refresh_count"),
                "acc_trigger_refresh_count": cache.get("acc_trigger_refresh_count"),
                "mean_psnr_db": quality.get("mean_psnr_db"),
                "mean_lpips": quality.get("mean_lpips"),
                "mean_mae": quality.get("mean_mae"),
                "mean_inference_time_seconds": cache.get("mean_inference_time_seconds"),
                "psnr_win_rate_over_baseline": quality.get("psnr_win_rate_over_baseline"),
                "lpips_win_rate_over_baseline": quality.get("lpips_win_rate_over_baseline"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summaries = root / "summaries"
    quality_csv = Path(args.quality_csv) if args.quality_csv else summaries / "final_pairwise_quality.csv"
    quality_summary_path = (
        Path(args.quality_summary)
        if args.quality_summary
        else summaries / "final_pairwise_quality_summary.json"
    )

    metadata = load_config_metadata(root)
    cache_rows = cache_summary_rows(root, metadata)
    quality_per_image = read_csv(quality_csv)
    quality_summary = load_json(quality_summary_path)
    improvement_rows = build_improvement_rows(quality_per_image)
    enriched_quality = add_improvement_summary(quality_summary, quality_per_image, improvement_rows)
    table_rows = final_table_rows(cache_rows, enriched_quality)

    write_json(summaries / "final_cache_summary.json", cache_rows)
    write_csv(summaries / "final_cache_summary.csv", cache_rows)
    write_json(summaries / "final_quality_summary.json", enriched_quality)
    write_csv(summaries / "final_quality_summary.csv", enriched_quality)
    write_csv(summaries / "final_paired_improvement.csv", improvement_rows)
    write_json(summaries / "final_summary_table.json", table_rows)
    write_csv(summaries / "final_summary_table.csv", table_rows)

    print(f"Wrote final cache summary: {summaries / 'final_cache_summary.json'}")
    print(f"Wrote final quality summary: {summaries / 'final_quality_summary.json'}")
    print(f"Wrote paired improvements: {summaries / 'final_paired_improvement.csv'}")
    print(f"Wrote final table: {summaries / 'final_summary_table.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select matched-refresh dual-gate configs from FLUX-dev calibration sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_ROOT = "experiments/fluxdev_150_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select formal dual-gate configs from calibration results.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Experiment root directory.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Calibration baseline cache summary. Defaults to ROOT/summaries/calib_baseline_acc030_cache.json.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output selected config JSON. Defaults to ROOT/configs/selected_dual_configs.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def reason_count(summary: Dict[str, Any], key: str) -> int:
    if f"{key}_trigger_refresh_count" in summary:
        return int(summary[f"{key}_trigger_refresh_count"])
    return int(summary.get("refresh_reason", {}).get(key, {}).get("count", 0))


def load_sweep_rows(root: Path) -> List[Dict[str, Any]]:
    summaries_dir = root / "summaries" / "calib_dual_sweep"
    configs_dir = root / "configs" / "calib_dual_sweep"
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(summaries_dir.glob("*_cache.json")):
        config_name = summary_path.name.removesuffix("_cache.json")
        config_path = configs_dir / f"{config_name}.json"
        if not config_path.exists():
            continue
        summary = load_json(summary_path)
        config = load_json(config_path)
        rows.append(
            {
                "config": config_name,
                "delta_single_quantile": config.get("delta_single_quantile"),
                "delta_single": as_float(config.get("delta_single")),
                "delta_acc": as_float(config.get("delta_acc")),
                "refresh_ratio": as_float(summary.get("refresh_ratio")),
                "refresh_steps": int(summary.get("num_refresh", 0)),
                "skip_steps": int(summary.get("num_skip", 0)),
                "single_trigger_refresh_count": reason_count(summary, "single"),
                "acc_trigger_refresh_count": reason_count(summary, "acc"),
                "warmup_refresh_count": int(summary.get("warmup_refresh_count", 0)),
                "summary_path": str(summary_path),
                "config_path": str(config_path),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def closest(rows: List[Dict[str, Any]], quantile: str, baseline_refresh_ratio: float) -> Dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("delta_single_quantile") == quantile and row.get("refresh_ratio") is not None
    ]
    if not candidates:
        raise ValueError(f"No calibration sweep rows found for {quantile}.")
    return min(
        candidates,
        key=lambda row: (
            abs(float(row["refresh_ratio"]) - baseline_refresh_ratio),
            float(row["refresh_ratio"]),
            float(row["delta_acc"]),
        ),
    )


def same_acc(rows: List[Dict[str, Any]], quantile: str, delta_acc: float) -> Dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("delta_single_quantile") == quantile
        and row.get("delta_acc") is not None
        and abs(float(row["delta_acc"]) - delta_acc) < 1e-9
    ]
    if not candidates:
        raise ValueError(f"No calibration sweep row found for {quantile} at delta_acc={delta_acc}.")
    return candidates[0]


def selected_record(row: Dict[str, Any], role: str, baseline_refresh_ratio: float) -> Dict[str, Any]:
    refresh_ratio = float(row["refresh_ratio"])
    return {
        "role": role,
        "source_calibration_config": row["config"],
        "delta_single_quantile": row["delta_single_quantile"],
        "delta_single": row["delta_single"],
        "delta_acc": row["delta_acc"],
        "calibration_refresh_ratio": refresh_ratio,
        "baseline_refresh_ratio": baseline_refresh_ratio,
        "abs_refresh_ratio_gap": abs(refresh_ratio - baseline_refresh_ratio),
        "single_trigger_refresh_count": row["single_trigger_refresh_count"],
        "acc_trigger_refresh_count": row["acc_trigger_refresh_count"],
        "skip_steps": row["skip_steps"],
        "refresh_steps": row["refresh_steps"],
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    baseline_path = Path(args.baseline) if args.baseline else root / "summaries" / "calib_baseline_acc030_cache.json"
    out_path = Path(args.out) if args.out else root / "configs" / "selected_dual_configs.json"

    baseline = load_json(baseline_path)
    baseline_refresh_ratio = float(baseline["refresh_ratio"])
    rows = load_sweep_rows(root)
    if not rows:
        raise ValueError(f"No calibration sweep summaries found under {root / 'summaries' / 'calib_dual_sweep'}.")

    q90_matched = closest(rows, "q90", baseline_refresh_ratio)
    q925_matched = closest(rows, "q925", baseline_refresh_ratio)
    q95_matched = closest(rows, "q95", baseline_refresh_ratio)
    q975_matched = closest(rows, "q975", baseline_refresh_ratio)
    q95_same_acc = same_acc(rows, "q95", 0.30)

    for row in rows:
        row["abs_refresh_ratio_gap"] = (
            None
            if row.get("refresh_ratio") is None
            else abs(float(row["refresh_ratio"]) - baseline_refresh_ratio)
        )

    write_csv(root / "summaries" / "calib_dual_sweep_selection.csv", rows)
    output = {
        "baseline_summary": str(baseline_path),
        "baseline_refresh_ratio": baseline_refresh_ratio,
        "selected": {
            "dual_q90_matched": selected_record(q90_matched, "matched", baseline_refresh_ratio),
            "dual_q925_matched": selected_record(q925_matched, "matched", baseline_refresh_ratio),
            "dual_q95_matched": selected_record(q95_matched, "matched", baseline_refresh_ratio),
            "dual_q975_matched": selected_record(q975_matched, "matched", baseline_refresh_ratio),
            "dual_q95_same_acc": selected_record(q95_same_acc, "same_acc", baseline_refresh_ratio),
        },
    }
    write_json(out_path, output)

    print(f"Wrote selected configs: {out_path}")
    for name, item in output["selected"].items():
        print(
            f"{name}: delta_acc={item['delta_acc']} delta_single={item['delta_single']} "
            f"calib_refresh={item['calibration_refresh_ratio']:.6f} "
            f"gap={item['abs_refresh_ratio_gap']:.6f}"
        )


if __name__ == "__main__":
    main()

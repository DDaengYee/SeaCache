#!/usr/bin/env python3
"""Analyze FLUX SeaCache cache decision JSONL logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REASON_KEYS = ("warmup", "acc", "single", "skip")
QUANTILES = (
    ("0.85", 0.85),
    ("0.90", 0.90),
    ("0.95", 0.95),
    ("0.975", 0.975),
    ("0.99", 0.99),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SeaCache cache decision JSONL logs.")
    parser.add_argument("--log", required=True, help="Path to cache decision JSONL log.")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Skip-distance quantile used as chosen_delta_single.",
    )
    parser.add_argument("--out", required=True, help="Path to write JSON analysis summary.")
    args = parser.parse_args()

    if not 0.0 <= args.quantile <= 1.0:
        parser.error("--quantile must be between 0 and 1.")
    return args


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def quantile(values: Iterable[float], q: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]

    pos = q * (len(ordered) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def csv_path_for(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_prompt_seed.csv")


def reason_summary(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    total = len(rows)
    counts = Counter(row.get("refresh_reason", "unknown") for row in rows)
    keys = list(REASON_KEYS)
    keys.extend(sorted(k for k in counts if k not in REASON_KEYS))
    return {
        key: {
            "count": int(counts.get(key, 0)),
            "ratio": (counts.get(key, 0) / total) if total else 0.0,
        }
        for key in keys
    }


def is_local_spike_skip(row: Dict[str, Any], chosen_delta_single: Optional[float]) -> bool:
    if chosen_delta_single is None:
        return False
    if as_bool(row.get("refresh")) is not False:
        return False
    distance = as_float(row.get("distance"))
    return distance is not None and distance > chosen_delta_single


def is_reset_spike(row: Dict[str, Any], chosen_delta_single: Optional[float]) -> bool:
    if not is_local_spike_skip(row, chosen_delta_single):
        return False
    steps_since_refresh = as_float(row.get("steps_since_refresh"))
    return steps_since_refresh is not None and steps_since_refresh <= 2


def group_key(row: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
    prompt_id = row.get("prompt_id")
    seed = row.get("seed")
    if prompt_id is None or seed is None:
        return None
    return prompt_id, seed


def prompt_seed_summary(
    rows: List[Dict[str, Any]], chosen_delta_single: Optional[float]
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = group_key(row)
        if key is not None:
            groups[key].append(row)

    summaries: List[Dict[str, Any]] = []
    for (prompt_id, seed), group_rows in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        num_steps = len(group_rows)
        num_refresh = sum(1 for row in group_rows if as_bool(row.get("refresh")) is True)
        summaries.append(
            {
                "prompt_id": prompt_id,
                "seed": seed,
                "num_steps": num_steps,
                "num_refresh": num_refresh,
                "refresh_ratio": (num_refresh / num_steps) if num_steps else 0.0,
                "local_spike_skip_count": sum(
                    1 for row in group_rows if is_local_spike_skip(row, chosen_delta_single)
                ),
                "reset_spike_count": sum(1 for row in group_rows if is_reset_spike(row, chosen_delta_single)),
            }
        )
    return summaries


def write_prompt_seed_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "prompt_id",
        "seed",
        "num_steps",
        "num_refresh",
        "refresh_ratio",
        "local_spike_skip_count",
        "reset_spike_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows: List[Dict[str, Any]], requested_quantile: float, log_path: Path) -> Dict[str, Any]:
    total_steps = len(rows)
    num_refresh = sum(1 for row in rows if as_bool(row.get("refresh")) is True)
    num_skip = sum(1 for row in rows if as_bool(row.get("refresh")) is False)
    refresh_reason = reason_summary(rows)

    skip_distances = [
        distance
        for row in rows
        if as_bool(row.get("refresh")) is False
        for distance in [as_float(row.get("distance"))]
        if distance is not None
    ]
    skip_distance_quantiles = {label: quantile(skip_distances, q) for label, q in QUANTILES}
    chosen_delta_single = quantile(skip_distances, requested_quantile)

    prompt_seed_rows = prompt_seed_summary(rows, chosen_delta_single)
    local_spike_skip_count = sum(1 for row in rows if is_local_spike_skip(row, chosen_delta_single))
    reset_after_refresh_spike_count = sum(1 for row in rows if is_reset_spike(row, chosen_delta_single))
    return {
        "log": str(log_path),
        "total_steps": total_steps,
        "num_refresh": num_refresh,
        "num_skip": num_skip,
        "refresh_ratio": (num_refresh / total_steps) if total_steps else 0.0,
        "refresh_reason": refresh_reason,
        "warmup_refresh_count": int(refresh_reason.get("warmup", {}).get("count", 0)),
        "acc_trigger_refresh_count": int(refresh_reason.get("acc", {}).get("count", 0)),
        "single_trigger_refresh_count": int(refresh_reason.get("single", {}).get("count", 0)),
        "skip_reason_count": int(refresh_reason.get("skip", {}).get("count", 0)),
        "skip_distance_quantiles": skip_distance_quantiles,
        "requested_quantile": requested_quantile,
        "chosen_delta_single": chosen_delta_single,
        "local_spike_skip_count": local_spike_skip_count,
        "post_refresh_spike_count": local_spike_skip_count,
        "reset_after_refresh_spike_count": reset_after_refresh_spike_count,
        "prompt_seed_summary": prompt_seed_rows,
    }


def print_summary(summary: Dict[str, Any], csv_out: Path) -> None:
    print("SeaCache cache log analysis")
    print(f"  total_steps: {summary['total_steps']}")
    print(f"  refresh_steps: {summary['num_refresh']}")
    print(f"  skip_steps: {summary['num_skip']}")
    print(f"  refresh_ratio: {summary['refresh_ratio']:.6f}")
    print(f"  chosen_delta_single(q={summary['requested_quantile']}): {summary['chosen_delta_single']}")
    print(f"  local_spike_skip_count: {summary['local_spike_skip_count']}")
    print(f"  reset_after_refresh_spike_count: {summary['reset_after_refresh_spike_count']}")
    print(f"  json_out: {summary['out']}")
    print(f"  csv_out: {csv_out}")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(log_path)
    summary = analyze(rows, args.quantile, log_path)
    summary["out"] = str(out_path)

    csv_out = csv_path_for(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    write_prompt_seed_csv(csv_out, summary["prompt_seed_summary"])
    print_summary(summary, csv_out)


if __name__ == "__main__":
    main()

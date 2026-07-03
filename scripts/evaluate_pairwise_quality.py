#!/usr/bin/env python3
"""Compute paired image fidelity metrics against a full-compute reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


IMAGE_RE = re.compile(r"SeaCache_p(?P<prompt_id>\d+)_seed(?P<seed>-?\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cached FLUX outputs against paired reference images.")
    parser.add_argument("--reference-dir", required=True, help="Directory containing full-refresh reference images.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate as name=directory. Can be repeated.",
    )
    parser.add_argument("--out", required=True, help="Path to write per-image CSV metrics.")
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS if the lpips package is installed.")
    parser.add_argument("--device", default=None, help="Torch device for LPIPS, e.g. cuda or cpu. Defaults automatically.")
    return parser.parse_args()


def parse_candidate(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--candidate must be name=directory, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"--candidate name cannot be empty: {value}")
    return name, Path(path)


def image_key(path: Path) -> Optional[Tuple[int, int]]:
    match = IMAGE_RE.search(path.name)
    if not match:
        return None
    return int(match.group("prompt_id")), int(match.group("seed"))


def index_images(directory: Path) -> Dict[Tuple[int, int], Path]:
    out: Dict[Tuple[int, int], Path] = {}
    for path in sorted(directory.glob("*.png")):
        key = image_key(path)
        if key is not None and key not in out:
            out[key] = path
    return out


def load_rgb01(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_pixel_metrics(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
    diff = candidate - reference
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = math.sqrt(mse)
    psnr = "inf" if mse == 0.0 else float(-10.0 * math.log10(mse))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr_db": psnr,
    }


def make_lpips_model(enabled: bool, device_arg: Optional[str]):
    if not enabled:
        return None, None
    try:
        import torch
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS requested, but dependency is missing. Install with: python -m pip install lpips") from exc

    device = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex").to(device).eval()
    return model, device


def lpips_distance(model, device: str, reference: np.ndarray, candidate: np.ndarray) -> float:
    import torch

    def to_tensor(image: np.ndarray):
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        return tensor.to(device)

    with torch.no_grad():
        value = model(to_tensor(reference), to_tensor(candidate))
    return float(value.detach().cpu().item())


def numeric_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(var)


def summarize(rows: List[Dict[str, Any]], missing: Dict[str, int]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)

    summaries = []
    for config, config_rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "config": config,
            "num_pairs": len(config_rows),
            "missing_pairs": missing.get(config, 0),
        }
        for metric in ("psnr_db", "mse", "rmse", "mae", "lpips"):
            values = numeric_values(config_rows, metric)
            mean, std = mean_std(values)
            summary[f"mean_{metric}"] = mean
            summary[f"std_{metric}"] = std
        summary["perfect_psnr_count"] = sum(1 for row in config_rows if row.get("psnr_db") == "inf")
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "config",
        "prompt_id",
        "seed",
        "psnr_db",
        "mse",
        "rmse",
        "mae",
        "lpips",
        "reference_image",
        "candidate_image",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    reference_dir = Path(args.reference_dir)
    candidates = [parse_candidate(value) for value in args.candidate]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reference_images = index_images(reference_dir)
    lpips_model, lpips_device = make_lpips_model(args.lpips, args.device)

    rows: List[Dict[str, Any]] = []
    missing: Dict[str, int] = {}
    for config_name, candidate_dir in candidates:
        candidate_images = index_images(candidate_dir)
        missing[config_name] = 0
        for key, candidate_path in sorted(candidate_images.items()):
            reference_path = reference_images.get(key)
            if reference_path is None:
                missing[config_name] += 1
                continue

            reference = load_rgb01(reference_path)
            candidate = load_rgb01(candidate_path)
            metrics = compute_pixel_metrics(reference, candidate)
            metrics["lpips"] = (
                lpips_distance(lpips_model, lpips_device, reference, candidate) if lpips_model is not None else None
            )
            prompt_id, seed = key
            rows.append(
                {
                    "config": config_name,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    **metrics,
                    "reference_image": str(reference_path),
                    "candidate_image": str(candidate_path),
                }
            )

    summary = summarize(rows, missing)
    summary_path = out_path.with_name(f"{out_path.stem}_summary.json")

    write_csv(out_path, rows)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Wrote per-image metrics: {out_path}")
    print(f"Wrote summary metrics: {summary_path}")
    for item in summary:
        psnr = item.get("mean_psnr_db")
        lpips_value = item.get("mean_lpips")
        print(
            f"{item['config']}: n={item['num_pairs']} "
            f"PSNR={psnr if psnr is not None else 'n/a'} "
            f"LPIPS={lpips_value if lpips_value is not None else 'n/a'}"
        )


if __name__ == "__main__":
    main()

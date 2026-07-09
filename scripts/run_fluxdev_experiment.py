#!/usr/bin/env python3
"""Run formal FLUX-dev SeaCache experiment stages.

This script is intentionally a thin orchestrator around FLUX/seacache_generate.py.
It does not modify the sampler, denoiser, SEA metric, or cache reuse logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_ROOT = "experiments/fluxdev_150_v1"
QUANTILE_KEYS = {"q85": "0.85", "q90": "0.90", "q925": "0.925", "q95": "0.95", "q975": "0.975"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX-dev SeaCache formal experiment stages.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Experiment root directory.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["calib-baseline", "calib-dual-sweep", "final-eval"],
        help="Experiment stage to run.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--force", action="store_true", help="Run even if metadata says a config is complete.")
    parser.add_argument("--only-config", default=None, help="Run only one config name from the selected stage.")
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip cache-log analysis after generation.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def jsonable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    for key, value in list(out.items()):
        if isinstance(value, Path):
            out[key] = str(value)
    return out


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def seeds_csv(root: Path) -> str:
    return ",".join(read_lines(root / "seeds.txt"))


def count_expected_images(prompt_file: Path, seeds_path: Path) -> int:
    return len(read_lines(prompt_file)) * len(read_lines(seeds_path))


def acc_tag(value: float) -> str:
    return f"acc{int(round(value * 100)):03d}"


def image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("SeaCache_p*_seed*.png")))


def command_to_text(command: Iterable[str]) -> str:
    return " ".join(str(item) for item in command)


def base_command(
    *,
    python: str,
    root: Path,
    settings: Dict[str, Any],
    prompt_file: Path,
    output_dir: Path,
    log_path: Path,
    cache_gate: str,
    delta_acc: float,
    delta_single: Optional[float],
    cache_disabled: bool = False,
) -> List[str]:
    command = [
        python,
        str(repo_root() / "FLUX" / "seacache_generate.py"),
        "--prompt-file",
        str(prompt_file),
        "--output_dir",
        str(output_dir),
        "--model_name",
        str(settings["model_name"]),
        "--width",
        str(settings["width"]),
        "--height",
        str(settings["height"]),
        "--num_inference_steps",
        str(settings["num_inference_steps"]),
        "--guidance",
        str(settings["guidance"]),
        "--dtype",
        str(settings["dtype"]),
        "--seeds",
        seeds_csv(root),
    ]
    if cache_disabled:
        command.append("--disable-seacache")
        return command

    command.extend(
        [
            "--seacache_thresh",
            str(delta_acc),
            "--cache-gate",
            cache_gate,
            "--log-cache-decisions",
            str(log_path),
        ]
    )
    if delta_single is not None:
        command.extend(["--delta-single", str(delta_single)])
    return command


def analysis_path(root: Path, stage: str, config_name: str) -> Path:
    if stage == "calib-baseline":
        return root / "summaries" / f"{config_name}_cache.json"
    if stage == "calib-dual-sweep":
        return root / "summaries" / "calib_dual_sweep" / f"{config_name}_cache.json"
    return root / "summaries" / "final_cache" / f"{config_name}_cache.json"


def config_path(root: Path, stage: str, config_name: str) -> Path:
    if stage == "calib-dual-sweep":
        return root / "configs" / "calib_dual_sweep" / f"{config_name}.json"
    if stage == "final-eval":
        return root / "configs" / "final" / f"{config_name}.json"
    return root / "configs" / f"{config_name}.json"


def load_calibration_quantiles(root: Path) -> Dict[str, float]:
    path = root / "summaries" / "calib_baseline_acc030_cache.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing calibration baseline analysis: {path}. "
            "Run --stage calib-baseline first."
        )
    summary = load_json(path)
    quantiles = summary.get("skip_distance_quantiles", {})
    out: Dict[str, float] = {}
    for label, json_key in QUANTILE_KEYS.items():
        value = quantiles.get(json_key)
        if value is None:
            raise ValueError(f"Calibration analysis is missing skip-distance quantile {json_key}.")
        out[label] = float(value)
    return out


def load_selected_configs(root: Path) -> Dict[str, Dict[str, Any]]:
    path = root / "configs" / "selected_dual_configs.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing selected dual configs: {path}. "
            "Run scripts/select_fluxdev_dual_configs.py after calibration sweep."
        )
    data = load_json(path)
    return data["selected"]


def calibration_baseline_configs(root: Path, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt_file = root / settings["calibration"]["prompts"]
    return [
        {
            "stage": "calib-baseline",
            "config": "calib_baseline_acc030",
            "prompt_file": prompt_file,
            "cache_gate": "accumulated",
            "delta_acc": float(settings["calibration"]["baseline_delta_acc"]),
            "delta_single": None,
            "delta_single_quantile": None,
        }
    ]


def calibration_sweep_configs(root: Path, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt_file = root / settings["calibration"]["prompts"]
    quantiles = load_calibration_quantiles(root)
    configs = []
    for q_label in settings["calibration"]["delta_single_quantiles"]:
        delta_single = quantiles[q_label]
        for delta_acc in settings["calibration"]["delta_acc_candidates"]:
            configs.append(
                {
                    "stage": "calib-dual-sweep",
                    "config": f"calib_dual_{q_label}_{acc_tag(float(delta_acc))}",
                    "prompt_file": prompt_file,
                    "cache_gate": "dual",
                    "delta_acc": float(delta_acc),
                    "delta_single": delta_single,
                    "delta_single_quantile": q_label,
                }
            )
    return configs


def final_eval_configs(root: Path, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt_file = root / settings["final_eval"]["prompts"]
    selected = load_selected_configs(root)
    configs = [
        {
            "stage": "final-eval",
            "config": "full_compute_reference",
            "prompt_file": prompt_file,
            "cache_gate": "accumulated",
            "delta_acc": 0.0,
            "delta_single": None,
            "delta_single_quantile": None,
            "cache_disabled": True,
            "reference_mode": "seacache_disabled",
        },
        {
            "stage": "final-eval",
            "config": "seacache_baseline_acc030",
            "prompt_file": prompt_file,
            "cache_gate": "accumulated",
            "delta_acc": float(settings["calibration"]["baseline_delta_acc"]),
            "delta_single": None,
            "delta_single_quantile": None,
        },
    ]
    if settings["final_eval"].get("include_dual_q90_matched", False):
        item = selected["dual_q90_matched"]
        configs.append(
            {
                "stage": "final-eval",
                "config": "dual_q90_matched",
                "prompt_file": prompt_file,
                "cache_gate": "dual",
                "delta_acc": float(item["delta_acc"]),
                "delta_single": float(item["delta_single"]),
                "delta_single_quantile": item["delta_single_quantile"],
            }
        )
    if settings["final_eval"].get("include_dual_q925_matched", False):
        item = selected["dual_q925_matched"]
        configs.append(
            {
                "stage": "final-eval",
                "config": "dual_q925_matched",
                "prompt_file": prompt_file,
                "cache_gate": "dual",
                "delta_acc": float(item["delta_acc"]),
                "delta_single": float(item["delta_single"]),
                "delta_single_quantile": item["delta_single_quantile"],
            }
        )
    for name in ("dual_q95_matched", "dual_q975_matched"):
        item = selected[name]
        configs.append(
            {
                "stage": "final-eval",
                "config": name,
                "prompt_file": prompt_file,
                "cache_gate": "dual",
                "delta_acc": float(item["delta_acc"]),
                "delta_single": float(item["delta_single"]),
                "delta_single_quantile": item["delta_single_quantile"],
            }
        )
    if settings["final_eval"].get("include_dual_q95_same_acc", True):
        item = selected["dual_q95_same_acc"]
        configs.append(
            {
                "stage": "final-eval",
                "config": "dual_q95_same_acc",
                "prompt_file": prompt_file,
                "cache_gate": "dual",
                "delta_acc": float(item["delta_acc"]),
                "delta_single": float(item["delta_single"]),
                "delta_single_quantile": item["delta_single_quantile"],
            }
        )
    return configs


def stage_configs(root: Path, settings: Dict[str, Any], stage: str) -> List[Dict[str, Any]]:
    if stage == "calib-baseline":
        return calibration_baseline_configs(root, settings)
    if stage == "calib-dual-sweep":
        return calibration_sweep_configs(root, settings)
    return final_eval_configs(root, settings)


def is_complete(config: Dict[str, Any], root: Path, expected_images: int) -> bool:
    metadata_path = root / "logs" / config["config"] / "run_metadata.json"
    output_dir = root / "outputs" / config["config"]
    if not metadata_path.exists():
        return False
    try:
        metadata = load_json(metadata_path)
    except json.JSONDecodeError:
        return False
    return metadata.get("status") == "complete" and image_count(output_dir) >= expected_images


def run_analysis(args: argparse.Namespace, root: Path, stage: str, config_name: str, log_path: Path) -> None:
    if args.skip_analysis:
        return
    out_path = analysis_path(root, stage, config_name)
    command = [
        args.python,
        str(repo_root() / "scripts" / "analyze_cache_logs.py"),
        "--log",
        str(log_path),
        "--quantile",
        "0.95",
        "--out",
        str(out_path),
    ]
    print(command_to_text(command))
    if not args.dry_run:
        subprocess.run(command, cwd=repo_root(), check=True)


def run_config(args: argparse.Namespace, root: Path, settings: Dict[str, Any], config: Dict[str, Any]) -> None:
    config_name = config["config"]
    stage = config["stage"]
    prompt_file = Path(config["prompt_file"])
    expected_images = count_expected_images(prompt_file, root / "seeds.txt")
    output_dir = root / "outputs" / config_name
    log_dir = root / "logs" / config_name
    log_path = log_dir / "cache_decisions.jsonl"
    run_metadata_path = log_dir / "run_metadata.json"
    run_command_path = log_dir / "run_command.txt"

    if not args.force and is_complete(config, root, expected_images):
        print(f"Skipping complete config: {config_name}")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = base_command(
        python=args.python,
        root=root,
        settings=settings,
        prompt_file=prompt_file,
        output_dir=output_dir,
        log_path=log_path,
        cache_gate=config["cache_gate"],
        delta_acc=float(config["delta_acc"]),
        delta_single=config.get("delta_single"),
        cache_disabled=bool(config.get("cache_disabled", False)),
    )

    config_record = {
        **jsonable_config(config),
        "expected_images": expected_images,
        "output_dir": str(output_dir),
        "log_path": None if config.get("cache_disabled", False) else str(log_path),
    }
    write_json(config_path(root, stage, config_name), config_record)
    run_command_path.write_text(command_to_text(command) + "\n", encoding="utf-8")

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] {config_name}")
    print(command_to_text(command))

    started = time.perf_counter()
    start_time = datetime.now().isoformat(timespec="seconds")
    status = "dry_run"
    if not args.dry_run:
        status = "running"
        write_json(
            run_metadata_path,
            {
                "config": config_name,
                "stage": stage,
                "status": status,
                "start_time": start_time,
                "command": command,
                "expected_images": expected_images,
            },
        )
        subprocess.run(command, cwd=repo_root(), check=True)
        status = "complete"
    elapsed = time.perf_counter() - started

    metadata = {
        "config": config_name,
        "stage": stage,
        "status": status,
        "start_time": start_time,
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "expected_images": expected_images,
        "observed_images": image_count(output_dir),
        "seconds_per_image": elapsed / expected_images if expected_images else None,
        "command": command,
        "config_path": str(config_path(root, stage, config_name)),
        "cache_log": None if config.get("cache_disabled", False) else str(log_path),
    }
    write_json(run_metadata_path, metadata)
    if not config.get("cache_disabled", False):
        run_analysis(args, root, stage, config_name, log_path)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    settings_path = root / "configs" / "base_settings.json"
    if not settings_path.exists():
        raise FileNotFoundError(
            f"Missing {settings_path}. Run scripts/prepare_fluxdev_experiment.py first."
        )
    settings = load_json(settings_path)
    configs = stage_configs(root, settings, args.stage)
    if args.only_config:
        configs = [config for config in configs if config["config"] == args.only_config]
        if not configs:
            raise ValueError(f"No config named {args.only_config!r} in stage {args.stage}.")

    print(f"Stage: {args.stage}")
    print(f"Configs: {len(configs)}")
    print(f"Root: {root}")
    for config in configs:
        run_config(args, root, settings, config)


if __name__ == "__main__":
    main()

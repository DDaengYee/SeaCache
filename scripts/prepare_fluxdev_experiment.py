#!/usr/bin/env python3
"""Prepare the formal FLUX-dev SeaCache experiment directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


DEFAULT_ROOT = "experiments/fluxdev_150_v1"

CALIB_PROMPTS = [
    "a natural light portrait of a textile designer with detailed hair and a linen jacket",
    "a close-up wildlife photo of a red fox in wet grass with detailed fur",
    "a mountain lake at sunrise with pine trees reflected in still water",
    "a narrow stone street in a coastal village with textured walls and soft shadows",
    "a cozy reading room with wooden shelves, fabric chairs, and warm table lamps",
    "a matte black mechanical keyboard on a clean desk beside a woven wrist rest",
    "a bowl of ramen with steam, scallions, sesame seeds, and glossy broth",
    "a fantasy crystal garden inside a moonlit cave with floating lanterns",
    "a minimal white ceramic vase on a pale blue studio background",
    "a macro photo of green leaves after rain covered with tiny droplets",
]

EVAL_PROMPTS = [
    "a studio portrait of a botanist wearing a green canvas apron, detailed hair, soft rim light",
    "an environmental portrait of a ceramic teacher in a bright clay workshop",
    "a close-up portrait of a street musician with a wool coat and textured scarf",
    "a profile portrait of a dancer with braided hair and soft motion blur",
    "a cinematic portrait of a polar expedition researcher in a simple white parka",
    "a golden retriever running through wet grass with detailed fur and water droplets",
    "a snow leopard standing on a rocky slope with windblown fur",
    "a hummingbird hovering near red flowers with crisp wing texture",
    "a tabby cat asleep on a woven blanket in window light",
    "koi fish swimming under clear pond water with ripples and reflections",
    "a wide desert landscape with layered dunes and long evening shadows",
    "an alpine meadow with wildflowers and distant snowy peaks",
    "a stormy ocean cliff with white foam and dark rocks",
    "an autumn forest path covered with dense red and yellow leaves",
    "a clear blue sky over a flat salt lake with a minimal horizon",
    "a modern glass library atrium with repeating steel beams",
    "a small brick cafe on a rainy street corner with pavement reflections",
    "a white concrete museum interior with a skylight and simple benches",
    "an old stone bridge over a narrow river in early morning fog",
    "a compact apartment kitchen with tile backsplash and hanging utensils",
    "an artist studio with paint tubes, brushes, canvas texture, and north light",
    "a quiet research lab bench with glassware, notebooks, and precise labels",
    "a sunlit bedroom with linen sheets, wooden floor, and soft curtains",
    "a greenhouse interior filled with dense plants and patterned shadows",
    "a brushed steel wristwatch resting on dark woven fabric",
    "a translucent perfume bottle on a reflective black surface",
    "a ceramic teapot and two cups on a wooden tray with steam",
    "a hiking backpack with zippers, buckles, and woven nylon straps",
    "wireless headphones on a minimal gray studio background",
    "a stack of pancakes with berries, syrup, and powdered sugar",
    "a sushi platter with rice texture, seaweed, ginger, and soy sauce",
    "a fresh bread loaf on a linen cloth with scattered flour",
    "a bowl of mixed fruit with glossy grapes and sliced citrus",
    "a chocolate cake slice with layered cream and fine crumbs",
    "a floating island fantasy landscape with waterfalls and morning clouds",
    "a distant castle-like observatory on a snowy mountain under stars",
    "a glowing underwater city with coral textures and soft blue light",
    "a dragon-shaped cloud formation above a moonlit valley",
    "an enchanted forest path with luminous mushrooms and mossy stones",
    "a celestial observatory room with brass instruments and star charts",
    "a macro photo of knitted fabric showing individual threads",
    "a close-up of braided hair strands decorated with tiny glass beads",
    "a field of tall grass in wind with sharp seed heads and sunlight",
    "a pond surface reflecting willow leaves with small circular ripples",
    "a colorful feather fan on a dark velvet background",
    "a simple red sphere on a seamless white studio background",
    "a single white sailboat under a clear blue sky with calm water",
    "a smooth clay cup on a plain table with one soft shadow",
    "a minimal studio shot of a blue cube against a light gray wall",
    "a foggy empty road disappearing into soft gradients",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the formal FLUX-dev experiment files.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Experiment root directory.")
    parser.add_argument(
        "--overwrite-prompts",
        action="store_true",
        help="Replace existing prompt and seed files. By default existing files are preserved.",
    )
    return parser.parse_args()


def write_lines(path: Path, lines: Iterable[str], overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")
    return True


def read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def validate_count(path: Path, expected: int) -> None:
    actual = len(read_nonempty_lines(path))
    if actual != expected:
        raise ValueError(f"{path} has {actual} non-empty lines; expected {expected}.")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    keep_dirs = {"configs/calib_dual_sweep", "configs/final", "logs", "outputs", "summaries"}
    for subdir in ("configs", "configs/calib_dual_sweep", "configs/final", "logs", "outputs", "summaries"):
        directory = root / subdir
        directory.mkdir(parents=True, exist_ok=True)
        if subdir in keep_dirs:
            (directory / ".gitkeep").touch()

    write_lines(root / "prompts_calib_10.txt", CALIB_PROMPTS, args.overwrite_prompts)
    write_lines(root / "prompts_eval_50.txt", EVAL_PROMPTS, args.overwrite_prompts)
    write_lines(root / "seeds.txt", ["0", "1", "2"], args.overwrite_prompts)

    validate_count(root / "prompts_calib_10.txt", 10)
    validate_count(root / "prompts_eval_50.txt", 50)
    validate_count(root / "seeds.txt", 3)

    settings = {
        "experiment": "fluxdev_150_v1",
        "model_name": "flux-dev",
        "num_inference_steps": 50,
        "width": 1024,
        "height": 1024,
        "guidance": 3.5,
        "dtype": "bf16",
        "seeds": [0, 1, 2],
        "calibration": {
            "prompts": "prompts_calib_10.txt",
            "baseline_delta_acc": 0.30,
            "delta_single_quantiles": ["q85", "q90", "q925", "q95", "q975"],
            "delta_acc_candidates": [0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42],
        },
        "final_eval": {
            "prompts": "prompts_eval_50.txt",
            "include_dual_q925_matched": False,
            "include_dual_q95_same_acc": True,
        },
        "notes": [
            "Thresholds must be computed from this calibration set, not from pilot runs.",
            "The generator is invoked once per config so the model is loaded once per config.",
        ],
    }
    write_json(root / "configs" / "base_settings.json", settings)

    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("logs/*\noutputs/*\n!logs/.gitkeep\n!outputs/.gitkeep\n", encoding="utf-8")

    print(f"Prepared experiment root: {root}")
    print("  calibration prompts: 10")
    print("  evaluation prompts: 50")
    print("  seeds: 0,1,2")
    print(f"  settings: {root / 'configs' / 'base_settings.json'}")


if __name__ == "__main__":
    main()

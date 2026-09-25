"""Aggregate behavior cloning and conservative fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate DoorKey imitation experiments")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/doorkey-imitation"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/23-doorkey-imitation.json")
    )
    args = parser.parse_args()
    bc = load(args.runs_root / "bc-8x8" / "history.json")
    naive = load(args.runs_root / "bc-ppo-8x8" / "history.json")
    conservative = load(args.runs_root / "bc-ppo-kl-8x8" / "history.json")
    report = {
        "experiment": "lesson-23-doorkey-imitation",
        "behavior_cloning": bc,
        "naive_finetune": naive,
        "reference_kl_finetune": conservative,
        "conclusion": (
            "Behavior cloning reaches high closed-loop success; unconstrained PPO forgets it, "
            "while reference KL preserves it."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "bc": bc["history"][-1],
                "naive": naive["evaluations"][-1],
                "reference_kl": conservative["evaluations"][-1],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

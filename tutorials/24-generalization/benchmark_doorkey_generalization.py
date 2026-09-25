"""Evaluate DoorKey agents on unseen sizes and seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doorkey_ppo import DoorKeyPPOAgent, evaluate_ppo


def build_report(runs_root: Path, episodes: int) -> dict[str, Any]:
    variants = {
        "ppo-5x5": runs_root / "doorkey-ppo" / "feedforward-seed-42" / "checkpoints" / "best.pt",
        "curriculum-8x8": runs_root
        / "doorkey-exploration"
        / "curriculum-v2-8x8"
        / "checkpoints"
        / "best.pt",
        "behavior-cloning": runs_root / "doorkey-imitation" / "bc-8x8" / "checkpoints" / "best.pt",
        "bc-reference-kl": runs_root
        / "doorkey-imitation"
        / "bc-ppo-kl-8x8"
        / "checkpoints"
        / "best.pt",
    }
    seeds = list(range(70_000, 70_000 + episodes))
    results = []
    for name, checkpoint_path in variants.items():
        agent = DoorKeyPPOAgent(recurrent=False)
        checkpoint = agent.load(checkpoint_path)
        for size in (5, 6, 8, 16):
            summary = evaluate_ppo(agent, seeds, size=size)
            results.append(
                {
                    "variant": name,
                    "checkpoint_step": checkpoint["steps"],
                    "size": size,
                    **summary,
                }
            )
    return {
        "experiment": "lesson-24-doorkey-generalization",
        "evaluation_seeds": seeds,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DoorKey generalization")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/24-doorkey-generalization.json")
    )
    args = parser.parse_args()
    report = build_report(args.runs_root, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

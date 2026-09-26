"""Evaluate raw and augmented PPO checkpoints under controlled pixel corruptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from carracing_env import DrivingConfig, VisualCarRacing, run_episode, summarize_episodes
from carracing_ppo import VisualPPOAgent


def evaluate(
    checkpoint: Path,
    *,
    corruption: str,
    severity: float,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    agent = VisualPPOAgent(4)
    step = agent.load(checkpoint)
    environment = VisualCarRacing(
        DrivingConfig(
            frame_stack=4,
            corruption=corruption,
            corruption_severity=severity,
        )
    )
    try:
        rows = [
            run_episode(environment, agent.predict, seed=seed + index) for index in range(episodes)
        ]
    finally:
        environment.close()
    return {"checkpoint_step": step, "summary": summarize_episodes(rows), "episodes": rows}


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--severity", type=float, default=0.6)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/30-visual-robustness.json")
    )
    args = parser.parse_args()
    checkpoints = {
        "raw_ppo": Path("runs/carracing-ppo/raw-seed-42/checkpoints/best.pt"),
        "augmented_ppo": Path("runs/carracing-ppo/augmented-seed-42/checkpoints/best.pt"),
    }
    corruptions = ("none", "brightness", "noise", "color", "blur", "occlusion")
    results = {
        agent_name: {
            corruption: evaluate(
                checkpoint,
                corruption=corruption,
                severity=0.0 if corruption == "none" else args.severity,
                episodes=args.episodes,
                seed=70_000,
            )
            for corruption in corruptions
        }
        for agent_name, checkpoint in checkpoints.items()
    }
    report = {
        "experiment": "30-visual-robustness",
        "episodes_per_condition": args.episodes,
        "severity": args.severity,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: {c: v["summary"]["average_return"] for c, v in rows.items()}
                for k, rows in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

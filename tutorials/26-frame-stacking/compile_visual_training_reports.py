"""Compile lessons 26-29 from training histories without hiding failed runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best(history: dict[str, Any]) -> dict[str, Any]:
    return max(history["evaluations"], key=lambda row: row["average_return"])


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    experiments = Path("docs/experiments")
    stack1 = load(Path("runs/carracing-dqn/stack1-seed-42/history.json"))
    stack4 = load(Path("runs/carracing-dqn/stack4-seed-42/history.json"))
    ppo_raw = load(Path("runs/carracing-ppo/raw-seed-42/history.json"))
    ppo_shaped = load(Path("runs/carracing-ppo/shaped-seed-42/history.json"))
    baselines = load(experiments / "25-visual-baselines.json")
    save(
        experiments / "26-frame-stacking.json",
        {
            "experiment": "26-frame-stacking",
            "budget_steps": 10_000,
            "variants": {
                "single_frame": {"best": best(stack1), "history": stack1},
                "four_frames": {"best": best(stack4), "history": stack4},
            },
        },
    )
    save(
        experiments / "27-discrete-dqn.json",
        {
            "experiment": "27-discrete-dqn",
            "action_count": 9,
            "best": best(stack4),
            "history": stack4,
            "baseline_summaries": baselines["summaries"],
        },
    )
    save(
        experiments / "28-continuous-ppo.json",
        {
            "experiment": "28-continuous-ppo",
            "variants": {
                "discrete_double_dqn": {"best": best(stack4), "history": stack4},
                "continuous_ppo": {"best": best(ppo_raw), "history": ppo_raw},
            },
        },
    )
    save(
        experiments / "29-reward-design.json",
        {
            "experiment": "29-reward-design",
            "variants": {
                "raw_reward": {"best": best(ppo_raw), "history": ppo_raw},
                "shaped_reward": {"best": best(ppo_shaped), "history": ppo_shaped},
            },
            "shaping": {
                "steering_penalty": 0.04,
                "brake_penalty": 0.08,
                "offroad_penalty": 0.15,
            },
        },
    )
    print("compiled lessons 26-29")


if __name__ == "__main__":
    main()

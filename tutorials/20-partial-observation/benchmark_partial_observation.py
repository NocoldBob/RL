"""Compare full and partial DoorKey DQN and quantify observation aliasing."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from doorkey_dqn import DoorKeyDQNAgent, evaluate_dqn
from doorkey_env import expert_plan, make_env
from minigrid.core.world_object import Key


def analyze_aliasing(episodes: int) -> dict[str, int | float]:
    env = make_env(8, observation_mode="compact_partial")
    groups: dict[bytes, dict[str, set[Any]]] = defaultdict(
        lambda: {"actions": set(), "has_key": set()}
    )
    samples = 0
    try:
        for seed in range(episodes):
            observation, _ = env.reset(seed=seed)
            plan = expert_plan(env)
            for action in plan:
                key = observation["image"].tobytes()
                groups[key]["actions"].add(action)
                groups[key]["has_key"].add(isinstance(env.unwrapped.carrying, Key))
                observation, _, terminated, truncated, _ = env.step(action)
                samples += 1
                if terminated or truncated:
                    break
    finally:
        env.close()
    repeated = [group for group in groups.values() if len(group["actions"]) > 1]
    stage_aliased = [group for group in groups.values() if len(group["has_key"]) > 1]
    return {
        "expert_samples": samples,
        "unique_local_observations": len(groups),
        "observations_with_multiple_expert_actions": len(repeated),
        "observations_seen_before_and_after_key": len(stage_aliased),
        "action_alias_rate": len(repeated) / len(groups),
    }


def build_report(runs_root: Path, episodes: int) -> dict[str, Any]:
    seeds = list(range(60_000, 60_000 + episodes))
    specifications = (
        (
            "full",
            runs_root / "doorkey-dqn" / "sparse-seed-42",
            "compact_full",
            5,
        ),
        (
            "partial",
            runs_root / "doorkey-partial-dqn" / "sparse-seed-42",
            "compact_partial",
            7,
        ),
    )
    variants = []
    histories = {}
    for name, run_dir, observation_mode, side in specifications:
        agent = DoorKeyDQNAgent(side, side)
        checkpoint = agent.load(run_dir / "checkpoints" / "best.pt")
        summary = evaluate_dqn(agent, 5, seeds, observation_mode=observation_mode)
        variants.append(
            {
                "observation_mode": name,
                "checkpoint_step": checkpoint["steps"],
                **summary,
            }
        )
        histories[name] = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    return {
        "experiment": "lesson-20-partial-observation-dqn",
        "training_seed": 42,
        "training_steps": 100_000,
        "evaluation_seeds": seeds,
        "variants": variants,
        "histories": histories,
        "aliasing": analyze_aliasing(500),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full and partial DoorKey DQN")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/20-doorkey-partial-dqn.json")
    )
    args = parser.parse_args()
    report = build_report(args.runs_root, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"variants": report["variants"], "aliasing": report["aliasing"]}, indent=2))


if __name__ == "__main__":
    main()

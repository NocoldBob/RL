"""Audit trained teams under unseen seeds, observation noise, and agent failure."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ppo_agents import MultiAgentPPO, PPOConfig
from spread_env import SpreadConfig, make_env, run_episode, summarize
from value_agents import MultiAgentValueLearner, ValueConfig


def value_policy(checkpoint: Path, mode: str, agent_id: bool = False):
    learner = MultiAgentValueLearner(ValueConfig(mode=mode, agent_id=agent_id))
    step = learner.load(checkpoint)
    return learner, step


def ppo_policy(checkpoint: Path, mode: str):
    learner = MultiAgentPPO(PPOConfig(mode=mode))
    step = learner.load(checkpoint)
    return learner, step


def condition_policy(learner, kind: str, rng: np.random.Generator):
    def policy(observations, environment):
        changed = {name: value.copy() for name, value in observations.items()}
        if kind == "noise":
            changed = {
                name: value + rng.normal(0, 0.1, value.shape).astype(np.float32)
                for name, value in changed.items()
            }
        actions = learner.policy(changed, environment)
        if kind == "agent_failure":
            actions["agent_0"] = 0
        return actions

    return policy


def evaluate(learner, episodes: int, kind: str) -> dict[str, Any]:
    environment = make_env(SpreadConfig())
    rng = np.random.default_rng(91_000)
    policy = condition_policy(learner, kind, rng)
    try:
        observations, _ = environment.reset(seed=91_000)
        started = time.perf_counter()
        for _ in range(100):
            policy(observations, environment)
        latency = 1000 * (time.perf_counter() - started) / 100
        rows = [run_episode(environment, policy, seed=91_000 + index) for index in range(episodes)]
    finally:
        environment.close()
    returns = np.asarray([row["team_return"] for row in rows])
    return {
        "summary": summarize(rows),
        "median_return": float(np.median(returns)),
        "p10_return": float(np.percentile(returns, 10)),
        "worst_return": float(returns.min()),
        "inference_ms": latency,
        "episodes": rows,
    }


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/39-marl-reliability.json")
    )
    args = parser.parse_args()
    specifications = {
        "iql": (value_policy, Path("runs/marl-value/iql/checkpoints/best.pt"), "iql", False),
        "qmix": (value_policy, Path("runs/marl-value/qmix/checkpoints/best.pt"), "qmix", True),
        "mappo": (ppo_policy, Path("runs/marl-ppo/mappo/checkpoints/best.pt"), "mappo", None),
        "communication": (
            ppo_policy,
            Path("runs/marl-ppo/communication/checkpoints/best.pt"),
            "communication",
            None,
        ),
    }
    results = {}
    for name, (loader, checkpoint, mode, agent_id) in specifications.items():
        learner, step = (
            loader(checkpoint, mode, agent_id) if agent_id is not None else loader(checkpoint, mode)
        )
        results[name] = {
            "checkpoint_step": step,
            "conditions": {
                kind: evaluate(learner, args.episodes, kind)
                for kind in ("clean", "noise", "agent_failure")
            },
        }
    report = {
        "experiment": "39-marl-reliability",
        "episodes_per_condition": args.episodes,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    kind: row["summary"]["average_team_return"]
                    for kind, row in payload["conditions"].items()
                }
                for name, payload in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

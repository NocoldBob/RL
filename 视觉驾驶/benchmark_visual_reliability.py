"""Final clean-track reliability audit for the visual-driving series."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from carracing_dqn import VisualDQNAgent
from carracing_env import DrivingConfig, VisualCarRacing, run_episode, summarize_episodes
from carracing_ppo import VisualPPOAgent


def load_agents() -> dict[str, tuple[Callable[[np.ndarray], int | np.ndarray], int]]:
    dqn = VisualDQNAgent(4, 9)
    dqn_step = dqn.load(Path("runs/carracing-dqn/stack4-seed-42/checkpoints/best.pt"))
    agents: dict[str, tuple[Callable[[np.ndarray], int | np.ndarray], int]] = {
        "discrete_dqn": (dqn.act, dqn_step)
    }
    for name, folder in (
        ("raw_ppo", "raw-seed-42"),
        ("shaped_ppo", "shaped-seed-42"),
        ("augmented_ppo", "augmented-seed-42"),
    ):
        agent = VisualPPOAgent(4)
        step = agent.load(Path("runs/carracing-ppo") / folder / "checkpoints" / "best.pt")
        agents[name] = (agent.predict, step)
    return agents


def audit(
    name: str,
    policy: Callable[[np.ndarray], int | np.ndarray],
    checkpoint_step: int,
    episodes: int,
) -> dict[str, Any]:
    environment = VisualCarRacing(
        DrivingConfig(
            frame_stack=4, action_mode="discrete" if name == "discrete_dqn" else "continuous"
        )
    )
    try:
        observation, _ = environment.reset(seed=90_000)
        started = time.perf_counter()
        for _ in range(100):
            policy(observation)
        inference_ms = 1000 * (time.perf_counter() - started) / 100
        rows = [run_episode(environment, policy, seed=90_000 + index) for index in range(episodes)]
    finally:
        environment.close()
    returns = np.asarray([row["return"] for row in rows], dtype=np.float32)
    return {
        "checkpoint_step": checkpoint_step,
        "summary": summarize_episodes(rows),
        "median_return": float(np.median(returns)),
        "worst_return": float(returns.min()),
        "p10_return": float(np.percentile(returns, 10)),
        "inference_ms": inference_ms,
        "episodes": rows,
    }


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/31-driving-reliability.json")
    )
    args = parser.parse_args()
    results = {
        name: audit(name, policy, step, args.episodes)
        for name, (policy, step) in load_agents().items()
    }
    report = {
        "experiment": "31-driving-reliability",
        "episodes_per_agent": args.episodes,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: row["summary"] for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()

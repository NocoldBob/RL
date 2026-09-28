"""Collect a reproducible mixture of random, smooth, and heuristic Pendulum behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dynamics_model import DynamicsDataset, write_json
from pendulum_env import heuristic_action, make_env


def collect(episodes: int = 120, seed: int = 42) -> DynamicsDataset:
    rng = np.random.default_rng(seed)
    env = make_env()
    fields: dict[str, list] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "episode_id": [],
    }
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            action = rng.uniform(-2.0, 2.0, size=1).astype(np.float32)
            terminated = truncated = False
            while not (terminated or truncated):
                mode = episode % 3
                if mode == 0:
                    action = rng.uniform(-2.0, 2.0, size=1).astype(np.float32)
                elif mode == 1:
                    action = np.clip(
                        0.85 * action + rng.normal(0.0, 0.45, size=1), -2.0, 2.0
                    ).astype(np.float32)
                else:
                    action = np.clip(
                        heuristic_action(observation) + rng.normal(0.0, 0.2, size=1), -2.0, 2.0
                    ).astype(np.float32)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                fields["observations"].append(observation)
                fields["actions"].append(action)
                fields["rewards"].append(reward)
                fields["next_observations"].append(next_observation)
                fields["episode_id"].append(episode)
                observation = next_observation
    finally:
        env.close()
    return DynamicsDataset(
        observations=np.asarray(fields["observations"], dtype=np.float32),
        actions=np.asarray(fields["actions"], dtype=np.float32),
        rewards=np.asarray(fields["rewards"], dtype=np.float32),
        next_observations=np.asarray(fields["next_observations"], dtype=np.float32),
        episode_id=np.asarray(fields["episode_id"], dtype=np.int64),
    )


def main() -> None:
    dataset = collect()
    path = Path("datasets/pendulum-model-based/transitions.npz")
    dataset.save(path)
    report = {
        "experiment": "49-dynamics-dataset",
        "environment": "Pendulum-v1",
        "collection": "equal random, smooth-random, and noisy heuristic episodes",
        "summary": dataset.summary(),
    }
    write_json(Path("docs/experiments/49-dynamics-dataset.json"), report)
    (path.parent / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

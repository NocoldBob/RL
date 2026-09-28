"""Collect random, medium, expert, and mixed Acrobot datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from dataset import OfflineDataset, concatenate, write_manifest
from online_dqn import DQNAgent


def epsilon_probability(chosen: int, greedy: int, epsilon: float) -> float:
    return epsilon / 3 + (1 - epsilon if chosen == greedy else 0.0)


def collect(
    episodes: int,
    seed: int,
    source_id: int,
    *,
    checkpoint: Path | None = None,
    epsilon: float = 1.0,
) -> OfflineDataset:
    torch.set_num_threads(1)
    environment = gym.make("Acrobot-v1")
    agent = DQNAgent()
    if checkpoint is not None:
        agent.load(checkpoint)
    data: dict[str, list] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "terminated": [],
        "truncated": [],
        "episode_id": [],
        "behavior_probability": [],
        "source": [],
    }
    rng = np.random.default_rng(seed)
    try:
        for episode in range(episodes):
            observation, _ = environment.reset(seed=seed + episode)
            done = False
            while not done:
                greedy = agent.act(observation) if checkpoint is not None else 0
                action = int(rng.integers(3)) if rng.random() < epsilon else greedy
                next_observation, reward, terminated, truncated, _ = environment.step(action)
                data["observations"].append(observation.astype(np.float32))
                data["actions"].append(action)
                data["rewards"].append(float(reward))
                data["next_observations"].append(next_observation.astype(np.float32))
                data["terminated"].append(terminated)
                data["truncated"].append(truncated)
                data["episode_id"].append(episode)
                data["behavior_probability"].append(
                    1 / 3 if checkpoint is None else epsilon_probability(action, greedy, epsilon)
                )
                data["source"].append(source_id)
                observation = next_observation
                done = terminated or truncated
    finally:
        environment.close()
    dtypes = {
        "observations": np.float32,
        "actions": np.int64,
        "rewards": np.float32,
        "next_observations": np.float32,
        "terminated": np.bool_,
        "truncated": np.bool_,
        "episode_id": np.int64,
        "behavior_probability": np.float32,
        "source": np.int8,
    }
    return OfflineDataset(
        **{name: np.asarray(values, dtype=dtypes[name]) for name, values in data.items()}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("runs/acrobot-online-dqn/seed-42/checkpoints")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/acrobot"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "random": collect(args.episodes, 10_000, 0),
        "medium": collect(
            args.episodes,
            20_000,
            1,
            checkpoint=args.checkpoint_dir / "step-50000.pt",
            epsilon=0.15,
        ),
        "expert": collect(
            args.episodes,
            30_000,
            2,
            checkpoint=args.checkpoint_dir / "best.pt",
            epsilon=0.02,
        ),
    }
    datasets["mixed"] = concatenate([datasets["random"], datasets["medium"], datasets["expert"]])
    for name, dataset in datasets.items():
        dataset.save(args.output_dir / f"{name}.npz")
    write_manifest(args.output_dir / "manifest.json", datasets)
    print(json.dumps({name: dataset.summary() for name, dataset in datasets.items()}, indent=2))


if __name__ == "__main__":
    main()

"""Acrobot evaluation helpers shared by online and offline agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

OBSERVATION_SIZE = 6
ACTION_COUNT = 3
MAX_STEPS = 500


def make_env():
    return gym.make("Acrobot-v1")


def run_episode(
    policy: Callable[[np.ndarray], int],
    *,
    seed: int,
    observation_noise: float = 0.0,
    action_sticky_probability: float = 0.0,
) -> dict[str, float | int | bool]:
    environment = make_env()
    rng = np.random.default_rng(seed)
    observation, _ = environment.reset(seed=seed)
    total_reward = 0.0
    previous_action = 1
    steps = 0
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            policy_observation = observation.astype(np.float32)
            if observation_noise:
                policy_observation = policy_observation + rng.normal(
                    0.0, observation_noise, policy_observation.shape
                ).astype(np.float32)
            action = int(policy(policy_observation))
            if steps and rng.random() < action_sticky_probability:
                action = previous_action
            observation, reward, terminated, truncated, _ = environment.step(action)
            previous_action = action
            total_reward += float(reward)
            steps += 1
    finally:
        environment.close()
    return {
        "seed": seed,
        "return": total_reward,
        "steps": steps,
        "success": bool(terminated),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    returns = np.asarray([row["return"] for row in rows], dtype=np.float32)
    steps = np.asarray([row["steps"] for row in rows], dtype=np.float32)
    successes = np.asarray([row["success"] for row in rows], dtype=np.float32)
    return {
        "episodes": float(len(rows)),
        "average_return": float(returns.mean()),
        "return_std": float(returns.std()),
        "average_steps": float(steps.mean()),
        "success_rate": float(successes.mean()),
        "median_steps": float(np.median(steps)),
        "p90_steps": float(np.percentile(steps, 90)),
    }


def evaluate_policy(
    policy: Callable[[np.ndarray], int],
    seeds: list[int],
    **conditions: float,
) -> dict[str, Any]:
    rows = [run_episode(policy, seed=seed, **conditions) for seed in seeds]
    return {"summary": summarize(rows), "episodes": rows}

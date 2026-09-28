"""Pendulum helpers and consistent model-based-control evaluation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

ENV_ID = "Pendulum-v1"
ACTION_LIMIT = 2.0


def make_env(*, gravity: float = 10.0, mass: float = 1.0, length: float = 1.0):
    env = gym.make(ENV_ID, g=gravity)
    env.unwrapped.m = mass
    env.unwrapped.l = length
    return env


def angle(observation: np.ndarray) -> float:
    return float(np.arctan2(observation[1], observation[0]))


def heuristic_action(observation: np.ndarray) -> np.ndarray:
    """Energy-shaped controller used only as an interpretable reference."""
    theta = angle(observation)
    velocity = float(observation[2])
    energy = 0.5 * velocity**2 + np.cos(theta) - 1.0
    swing = np.sign(velocity * np.cos(theta) + 1e-6) * (1.2 - 0.5 * energy)
    stabilize = -3.0 * theta - 0.8 * velocity
    action = stabilize if abs(theta) < 0.45 else swing
    return np.asarray([np.clip(action, -ACTION_LIMIT, ACTION_LIMIT)], dtype=np.float32)


def evaluate_controller(
    controller: Callable[[np.ndarray], np.ndarray],
    seeds: list[int],
    *,
    gravity: float = 10.0,
    mass: float = 1.0,
    length: float = 1.0,
    observation_noise: float = 0.0,
    action_delay: int = 0,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        env = make_env(gravity=gravity, mass=mass, length=length)
        observation, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 991)
        delayed = deque(
            [np.zeros(1, dtype=np.float32)] * (action_delay + 1), maxlen=action_delay + 1
        )
        total = 0.0
        upright = 0
        action_energy = 0.0
        terminated = truncated = False
        try:
            while not (terminated or truncated):
                perceived = observation + rng.normal(0.0, observation_noise, 3)
                action = np.asarray(controller(perceived), dtype=np.float32).reshape(1)
                action = np.clip(action, -ACTION_LIMIT, ACTION_LIMIT)
                delayed.append(action)
                executed = delayed[0]
                observation, reward, terminated, truncated, _ = env.step(executed)
                total += float(reward)
                action_energy += float(executed[0] ** 2)
                upright += int(abs(angle(observation)) < 0.2 and abs(observation[2]) < 1.0)
        finally:
            env.close()
        rows.append(
            {
                "seed": seed,
                "return": total,
                "upright_fraction": upright / 200.0,
                "action_energy": action_energy,
            }
        )
    returns = np.asarray([row["return"] for row in rows])
    return {
        "summary": {
            "episodes": len(rows),
            "average_return": float(returns.mean()),
            "return_std": float(returns.std()),
            "p10_return": float(np.percentile(returns, 10)),
            "worst_return": float(returns.min()),
            "upright_fraction": float(np.mean([row["upright_fraction"] for row in rows])),
            "average_action_energy": float(np.mean([row["action_energy"] for row in rows])),
        },
        "episodes": rows,
    }

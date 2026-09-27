"""MPE Simple Spread helpers, privileged baselines, and common metrics."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from mpe2 import simple_spread_v3


@dataclass(slots=True)
class SpreadConfig:
    agents: int = 3
    max_cycles: int = 25
    local_ratio: float = 0.5
    continuous_actions: bool = False


def make_env(config: SpreadConfig):
    return simple_spread_v3.parallel_env(
        N=config.agents,
        max_cycles=config.max_cycles,
        local_ratio=config.local_ratio,
        continuous_actions=config.continuous_actions,
    )


def agent_index(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def world_positions(environment) -> tuple[np.ndarray, np.ndarray]:
    world = environment.unwrapped.world
    agents = np.asarray([agent.state.p_pos for agent in world.agents], dtype=np.float32)
    landmarks = np.asarray([landmark.state.p_pos for landmark in world.landmarks], dtype=np.float32)
    return agents, landmarks


def assignment(agents: np.ndarray, landmarks: np.ndarray) -> tuple[int, ...]:
    """Brute-force minimum assignment; intended for the small teaching environment."""
    if len(agents) != len(landmarks):
        raise ValueError("agent and landmark counts must match")
    return min(
        itertools.permutations(range(len(landmarks))),
        key=lambda order: sum(
            np.linalg.norm(agents[i] - landmarks[j]) for i, j in enumerate(order)
        ),
    )


def direction_action(delta: np.ndarray) -> int:
    """MPE discrete actions: no-op, left, right, down, up."""
    if np.linalg.norm(delta) < 0.08:
        return 0
    if abs(float(delta[0])) > abs(float(delta[1])):
        return 2 if delta[0] > 0 else 1
    return 4 if delta[1] > 0 else 3


def privileged_actions(environment, *, coordinated: bool) -> dict[str, int]:
    agents, landmarks = world_positions(environment)
    if coordinated:
        targets = assignment(agents, landmarks)
    else:
        targets = tuple(
            int(np.argmin(np.linalg.norm(landmarks - position, axis=1))) for position in agents
        )
    return {
        name: direction_action(landmarks[targets[index]] - agents[index])
        for index, name in enumerate(environment.agents)
    }


def coverage_metrics(environment, threshold: float = 0.15) -> dict[str, float]:
    agents, landmarks = world_positions(environment)
    distances = np.linalg.norm(landmarks[:, None, :] - agents[None, :, :], axis=-1)
    nearest = distances.min(axis=1)
    collisions = 0
    for first, second in itertools.combinations(agents, 2):
        collisions += int(np.linalg.norm(first - second) < 0.2)
    return {
        "coverage": float(np.mean(nearest < threshold)),
        "full_coverage": float(np.all(nearest < threshold)),
        "mean_landmark_distance": float(nearest.mean()),
        "collisions": float(collisions),
    }


def run_episode(
    environment,
    policy: Callable[[dict[str, np.ndarray], Any], dict[str, int]],
    *,
    seed: int,
) -> dict[str, float | int]:
    observations, _ = environment.reset(seed=seed)
    total_reward = 0.0
    collision_steps = 0.0
    steps = 0
    metrics = coverage_metrics(environment)
    while environment.agents:
        actions = policy(observations, environment)
        observations, rewards, _, _, _ = environment.step(actions)
        total_reward += float(np.mean(list(rewards.values())))
        metrics = coverage_metrics(environment)
        collision_steps += metrics["collisions"]
        steps += 1
    return {
        "seed": seed,
        "team_return": total_reward,
        "coverage": metrics["coverage"],
        "full_coverage": metrics["full_coverage"],
        "mean_landmark_distance": metrics["mean_landmark_distance"],
        "collision_rate": collision_steps / max(1, steps),
        "steps": steps,
    }


def summarize(rows: list[dict[str, float | int]]) -> dict[str, float]:
    keys = (
        "team_return",
        "coverage",
        "full_coverage",
        "mean_landmark_distance",
        "collision_rate",
        "steps",
    )
    return {f"average_{key}": float(np.mean([float(row[key]) for row in rows])) for key in keys}


def random_policy(observations: dict[str, np.ndarray], environment) -> dict[str, int]:
    return {name: environment.action_space(name).sample() for name in observations}


def nearest_policy(_observations: dict[str, np.ndarray], environment) -> dict[str, int]:
    return privileged_actions(environment, coordinated=False)


def assignment_policy(_observations: dict[str, np.ndarray], environment) -> dict[str, int]:
    return privileged_actions(environment, coordinated=True)

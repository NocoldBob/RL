"""Baseline policies and evaluation helpers for continuous Mountain Car."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

ENV_ID = "MountainCarContinuous-v0"
POLICY_NAMES = ("zero", "random", "bang_bang", "smooth_momentum")
POLICY_LABELS = {
    "zero": "Zero throttle",
    "random": "Random throttle",
    "bang_bang": "Full-throttle momentum",
    "smooth_momentum": "Smooth momentum",
}


@dataclass
class EpisodeResult:
    policy: str
    seed: int
    episode_return: float
    success: bool
    steps: int
    action_energy: float
    max_position: float
    final_position: float
    trace: dict[str, list[float]] | None = None

    def metrics(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "episode_return": self.episode_return,
            "success": self.success,
            "steps": self.steps,
            "action_energy": self.action_energy,
            "max_position": self.max_position,
            "final_position": self.final_position,
        }


def _action(value: float) -> np.ndarray:
    return np.asarray([np.clip(value, -1.0, 1.0)], dtype=np.float32)


def select_action(
    policy: str,
    observation: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one bounded continuous action for a named baseline."""
    if policy == "zero":
        return _action(0.0)
    if policy == "random":
        return _action(float(rng.uniform(-1.0, 1.0)))

    velocity = float(observation[1])
    direction = 1.0 if velocity > 0.0 else -1.0
    if policy == "bang_bang":
        return _action(direction)
    if policy == "smooth_momentum":
        normalized_speed = min(abs(velocity) / 0.07, 1.0)
        throttle = 0.45 + 0.55 * normalized_speed
        return _action(direction * throttle)
    raise ValueError(f"unknown policy {policy!r}; expected one of {POLICY_NAMES}")


def run_episode(
    policy: str,
    seed: int,
    *,
    capture_trace: bool = False,
    render_mode: str | None = None,
) -> EpisodeResult:
    """Evaluate one baseline without training."""
    env = gym.make(ENV_ID, render_mode=render_mode)
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed)
    positions = [float(observation[0])]
    velocities = [float(observation[1])]
    actions: list[float] = []
    rewards: list[float] = []
    episode_return = 0.0
    action_energy = 0.0
    max_position = float(observation[0])
    terminated = False
    truncated = False
    steps = 0
    try:
        while not (terminated or truncated):
            action = select_action(policy, observation, rng)
            observation, reward, terminated, truncated, _ = env.step(action)
            action_value = float(action[0])
            steps += 1
            episode_return += float(reward)
            action_energy += action_value**2
            max_position = max(max_position, float(observation[0]))
            if capture_trace:
                positions.append(float(observation[0]))
                velocities.append(float(observation[1]))
                actions.append(action_value)
                rewards.append(float(reward))
    finally:
        env.close()

    trace = None
    if capture_trace:
        trace = {
            "position": positions,
            "velocity": velocities,
            "action": actions,
            "reward": rewards,
        }
    return EpisodeResult(
        policy=policy,
        seed=seed,
        episode_return=episode_return,
        success=terminated,
        steps=steps,
        action_energy=action_energy,
        max_position=max_position,
        final_position=float(observation[0]),
        trace=trace,
    )


def summarize(results: list[EpisodeResult]) -> dict[str, float | int | None]:
    returns = np.asarray([row.episode_return for row in results], dtype=np.float64)
    successes = np.asarray([row.success for row in results], dtype=np.float64)
    steps = np.asarray([row.steps for row in results], dtype=np.float64)
    energies = np.asarray([row.action_energy for row in results], dtype=np.float64)
    success_steps = [row.steps for row in results if row.success]
    return {
        "episodes": len(results),
        "average_return": float(returns.mean()),
        "return_std": float(returns.std()),
        "success_rate": float(successes.mean()),
        "average_steps": float(steps.mean()),
        "average_success_steps": (float(np.mean(success_steps)) if success_steps else None),
        "average_action_energy": float(energies.mean()),
        "average_action_cost": float(0.1 * energies.mean()),
    }


def evaluate_baselines(
    seeds: list[int],
    policies: tuple[str, ...] = POLICY_NAMES,
) -> tuple[dict[str, dict[str, float | int | None]], list[EpisodeResult]]:
    rows: list[EpisodeResult] = []
    aggregate: dict[str, dict[str, float | int | None]] = {}
    for policy in policies:
        policy_rows = [run_episode(policy, seed) for seed in seeds]
        rows.extend(policy_rows)
        aggregate[policy] = summarize(policy_rows)
    return aggregate, rows

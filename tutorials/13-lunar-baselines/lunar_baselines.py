"""Rule and random baselines for LunarLanderContinuous-v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

ENV_ID = "LunarLanderContinuous-v3"
POLICY_NAMES = ("zero", "random", "heuristic", "noisy_heuristic")


@dataclass(slots=True)
class EpisodeResult:
    policy: str
    seed: int
    episode_return: float
    steps: int
    landed: bool
    crashed: bool
    timed_out: bool
    main_engine_usage: float
    side_engine_usage: float
    final_x: float
    final_y: float
    final_speed: float
    final_angle: float

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def engine_power(action: np.ndarray) -> tuple[float, float]:
    """Reproduce the environment's two continuous engine activation rules."""
    main = (float(np.clip(action[0], 0.0, 1.0)) + 1.0) * 0.5 if action[0] > 0 else 0.0
    side = float(np.clip(abs(action[1]), 0.5, 1.0)) if abs(action[1]) > 0.5 else 0.0
    return main, side


def select_action(
    policy: str,
    env: gym.Env,
    observation: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if policy == "zero":
        action = np.zeros(2, dtype=np.float32)
    elif policy == "random":
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    elif policy in {"heuristic", "noisy_heuristic"}:
        action = np.asarray(heuristic(env, observation), dtype=np.float32)
        if policy == "noisy_heuristic":
            action = np.clip(action + rng.normal(0.0, 0.15, size=2), -1.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"unknown policy: {policy}")
    return action


def run_episode(
    policy: str,
    seed: int,
    *,
    render_mode: str | None = None,
    capture_trace: bool = False,
) -> tuple[EpisodeResult, dict[str, list[Any]] | None]:
    env = gym.make(ENV_ID, render_mode=render_mode)
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    episode_return = 0.0
    main_usage = 0.0
    side_usage = 0.0
    terminated = truncated = False
    last_reward = 0.0
    steps = 0
    trace = (
        {
            "x": [float(observation[0])],
            "y": [float(observation[1])],
            "x_velocity": [float(observation[2])],
            "y_velocity": [float(observation[3])],
            "angle": [float(observation[4])],
            "main_action": [],
            "side_action": [],
            "reward": [],
        }
        if capture_trace
        else None
    )
    try:
        while not (terminated or truncated):
            action = select_action(policy, env, observation, rng)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            main, side = engine_power(action)
            main_usage += main
            side_usage += side
            episode_return += float(reward)
            last_reward = float(reward)
            steps += 1
            if trace is not None:
                trace["x"].append(float(next_observation[0]))
                trace["y"].append(float(next_observation[1]))
                trace["x_velocity"].append(float(next_observation[2]))
                trace["y_velocity"].append(float(next_observation[3]))
                trace["angle"].append(float(next_observation[4]))
                trace["main_action"].append(float(action[0]))
                trace["side_action"].append(float(action[1]))
                trace["reward"].append(float(reward))
            observation = next_observation
    finally:
        env.close()

    landed = bool(terminated and last_reward == 100.0)
    crashed = bool(terminated and last_reward == -100.0)
    result = EpisodeResult(
        policy=policy,
        seed=seed,
        episode_return=episode_return,
        steps=steps,
        landed=landed,
        crashed=crashed,
        timed_out=bool(truncated),
        main_engine_usage=main_usage,
        side_engine_usage=side_usage,
        final_x=float(observation[0]),
        final_y=float(observation[1]),
        final_speed=float(np.hypot(observation[2], observation[3])),
        final_angle=float(observation[4]),
    )
    return result, trace

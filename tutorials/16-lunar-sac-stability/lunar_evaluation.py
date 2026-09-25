"""Shared deterministic evaluation for LunarLander agents."""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from lunar_baselines import ENV_ID, engine_power


class PredictAgent(Protocol):
    def predict(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray: ...


def evaluate_lunar_agent(
    agent: PredictAgent,
    seeds: list[int],
    *,
    capture_seed: int | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]], dict[str, list[Any]] | None]:
    env = gym.make(ENV_ID)
    rows = []
    captured_trace = None
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            episode_return = 0.0
            last_reward = 0.0
            steps = 0
            main_usage = side_usage = 0.0
            trace = (
                {
                    "x": [float(observation[0])],
                    "y": [float(observation[1])],
                    "main_action": [],
                    "side_action": [],
                    "reward": [],
                }
                if seed == capture_seed
                else None
            )
            while not (terminated or truncated):
                action = agent.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                main, side = engine_power(action)
                main_usage += main
                side_usage += side
                episode_return += float(reward)
                last_reward = float(reward)
                steps += 1
                if trace is not None:
                    trace["x"].append(float(observation[0]))
                    trace["y"].append(float(observation[1]))
                    trace["main_action"].append(float(action[0]))
                    trace["side_action"].append(float(action[1]))
                    trace["reward"].append(float(reward))
            landed = bool(terminated and last_reward == 100.0)
            rows.append(
                {
                    "seed": seed,
                    "episode_return": episode_return,
                    "landed": landed,
                    "crashed": bool(terminated and last_reward == -100.0),
                    "timed_out": bool(truncated),
                    "solved": episode_return >= 200.0,
                    "steps": steps,
                    "main_engine_usage": main_usage,
                    "side_engine_usage": side_usage,
                }
            )
            if trace is not None:
                captured_trace = trace
    finally:
        env.close()

    summary = {
        "episodes": len(rows),
        "average_return": float(np.mean([row["episode_return"] for row in rows])),
        "return_std": float(np.std([row["episode_return"] for row in rows])),
        "landing_rate": float(np.mean([row["landed"] for row in rows])),
        "crash_rate": float(np.mean([row["crashed"] for row in rows])),
        "solved_rate": float(np.mean([row["solved"] for row in rows])),
        "average_steps": float(np.mean([row["steps"] for row in rows])),
        "average_main_engine_usage": float(np.mean([row["main_engine_usage"] for row in rows])),
        "average_side_engine_usage": float(np.mean([row["side_engine_usage"] for row in rows])),
    }
    return summary, rows, captured_trace

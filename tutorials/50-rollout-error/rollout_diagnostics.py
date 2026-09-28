"""Measure how one-step dynamics errors compound over open-loop rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from dynamics_model import load_model, write_json
from pendulum_env import make_env


def main() -> None:
    torch.set_num_threads(1)
    model, metadata = load_model(Path("runs/model-based/dynamics/model-0.pt"))
    horizons = (1, 5, 10, 25, 50)
    errors = {horizon: [] for horizon in horizons}
    reward_errors = {horizon: [] for horizon in horizons}
    rng = np.random.default_rng(2026)
    for episode in range(40):
        env = make_env()
        observation, _ = env.reset(seed=70_000 + episode)
        actions = rng.uniform(-2.0, 2.0, size=(max(horizons), 1)).astype(np.float32)
        real_states = []
        real_rewards = []
        try:
            for action in actions:
                observation, reward, _, _, _ = env.step(action)
                real_states.append(observation.copy())
                real_rewards.append(float(reward))
        finally:
            env.close()
        # Recreate the exact starting state without relying on private environment state.
        env = make_env()
        starting, _ = env.reset(seed=70_000 + episode)
        env.close()
        predicted = torch.as_tensor(starting, dtype=torch.float32).unsqueeze(0)
        cumulative_reward = 0.0
        for step, action in enumerate(actions, start=1):
            predicted, reward = model.predict(predicted, torch.as_tensor(action).reshape(1, 1))
            cumulative_reward += float(reward[0])
            if step in horizons:
                errors[step].append(
                    float(np.linalg.norm(predicted[0].numpy() - real_states[step - 1]))
                )
                reward_errors[step].append(abs(cumulative_reward - sum(real_rewards[:step])))
    rows = [
        {
            "horizon": horizon,
            "state_rmse": float(np.sqrt(np.mean(np.square(errors[horizon])))),
            "reward_mae": float(np.mean(reward_errors[horizon])),
        }
        for horizon in horizons
    ]
    report = {"experiment": "50-rollout-error", "checkpoint": metadata, "horizons": rows}
    write_json(Path("docs/experiments/50-rollout-error.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

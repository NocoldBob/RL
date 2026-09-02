"""Render a deterministic continuous PPO episode."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
from continuous_ppo import ContinuousPPOAgent
from mountain_car_baselines import ENV_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Play continuous PPO")
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=Path("runs/continuous-ppo/seed-42/checkpoints/best.pt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    agent = ContinuousPPOAgent()
    agent.load_checkpoint(args.checkpoint)
    env = gym.make(ENV_ID, render_mode="human")
    observation, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    try:
        while True:
            action = agent.predict(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            time.sleep(max(0.0, 1.0 / args.fps))
            if terminated or truncated:
                print(f"reward={total_reward:.2f} success={terminated}")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()

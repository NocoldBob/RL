"""Render a deterministic SAC episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from mountain_car_baselines import ENV_ID
from sac import SACAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained SAC policy")
    parser.add_argument(
        "checkpoint", nargs="?", type=Path, default=Path("runs/sac/seed-42/checkpoints/best.pt")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    agent = SACAgent()
    agent.load_checkpoint(args.checkpoint)
    env = gym.make(ENV_ID, render_mode="human")
    observation, _ = env.reset(seed=args.seed)
    terminated = truncated = False
    episode_return = 0.0
    try:
        while not (terminated or truncated):
            action = agent.predict(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            env.render()
            if args.fps > 0:
                import time

                time.sleep(1.0 / args.fps)
    finally:
        env.close()
    print(f"return={episode_return:.2f} success={terminated}")


if __name__ == "__main__":
    main()

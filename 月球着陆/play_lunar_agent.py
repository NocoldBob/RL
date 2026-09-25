"""Render a trained LunarLander PPO or SAC policy."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
from lunar_agents import LunarPPOAgent, LunarSACAgent
from lunar_baselines import ENV_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained LunarLander agent")
    parser.add_argument("algorithm", choices=("ppo", "sac"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    agent = LunarPPOAgent() if args.algorithm == "ppo" else LunarSACAgent()
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
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
    finally:
        env.close()
    print(f"return={episode_return:.2f} terminated={terminated} truncated={truncated}")


if __name__ == "__main__":
    main()

"""Watch one hand-written baseline control MountainCarContinuous-v0."""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
from mountain_car_baselines import ENV_ID, POLICY_NAMES, select_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a continuous Mountain Car baseline")
    parser.add_argument("policy", choices=POLICY_NAMES, nargs="?", default="smooth_momentum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    arguments = parser.parse_args()

    env = gym.make(ENV_ID, render_mode="human")
    rng = np.random.default_rng(arguments.seed)
    observation, _ = env.reset(seed=arguments.seed)
    episode_return = 0.0
    steps = 0
    terminated = False
    truncated = False
    try:
        while not (terminated or truncated):
            action = select_action(arguments.policy, observation, rng)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            steps += 1
            if arguments.fps > 0:
                time.sleep(1.0 / arguments.fps)
    finally:
        env.close()
    print(
        f"policy={arguments.policy} return={episode_return:.2f} steps={steps} success={terminated}"
    )


if __name__ == "__main__":
    main()

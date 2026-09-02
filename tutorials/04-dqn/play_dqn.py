"""Render a trained DQN Snake agent without exploration."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from benchmark import load_dqn
from environment import SnakeEnv
from main import choose_device


def play_dqn(
    model_path: Path,
    *,
    device_name: str = "auto",
    seed: int = 42,
    fps: int = 15,
) -> None:
    device = choose_device(device_name)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = dict(checkpoint.get("config", {}))
    env = SnakeEnv(
        grid_size=int(config.get("grid_size", 6)),
        end_score=int(config.get("end_score", 4)),
        max_steps=int(config.get("max_steps", 100)),
        render_mode="human",
    )
    agent = load_dqn(model_path, device)
    random_source = random.Random(seed)
    observation, _ = env.reset(seed=seed)
    try:
        while not env.closed:
            action = agent.select_action(
                observation,
                epsilon=0.0,
                random_source=random_source,
            )
            observation, _, terminated, truncated, info = env.step(action)
            env.render(fps=fps)
            if terminated or truncated:
                print(
                    f"episode finished: score={info['score']} "
                    f"steps={info['steps']} terminated={terminated}"
                )
                break
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained Snake DQN checkpoint")
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=Path("runs/dqn/checkpoints/best.pt"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    arguments = parser.parse_args()
    play_dqn(
        arguments.checkpoint,
        device_name=arguments.device,
        seed=arguments.seed,
        fps=arguments.fps,
    )


if __name__ == "__main__":
    main()

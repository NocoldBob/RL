"""Render a trained PPO Snake agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from benchmark import load_ppo
from environment import SnakeEnv
from main import choose_device


def play_ppo(
    model_path: Path,
    *,
    device_name: str = "auto",
    seed: int = 42,
    fps: int = 15,
    deterministic: bool = False,
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
    agent = load_ppo(model_path, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    observation, _ = env.reset(seed=seed)
    try:
        while not env.closed:
            action = agent.predict(
                observation,
                deterministic=deterministic,
                generator=generator,
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
    parser = argparse.ArgumentParser(description="Play a trained Snake PPO checkpoint")
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=Path("runs/ppo/checkpoints/best.pt"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--deterministic", action="store_true")
    arguments = parser.parse_args()
    play_ppo(
        arguments.checkpoint,
        device_name=arguments.device,
        seed=arguments.seed,
        fps=arguments.fps,
        deterministic=arguments.deterministic,
    )


if __name__ == "__main__":
    main()

"""Render a trained Snake agent without teacher assistance."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from environment import SnakeEnv
from main import as_tensor, choose_device
from model import ConvActorCritic


def play_game(
    model_path: Path,
    *,
    device_name: str = "auto",
    seed: int = 42,
    fps: int = 15,
) -> None:
    device = choose_device(device_name)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = dict(checkpoint.get("config", {}))
    grid_size = int(config.get("grid_size", 6))
    end_score = int(config.get("end_score", 4))
    max_steps = int(config.get("max_steps", 100))

    env = SnakeEnv(
        grid_size=grid_size,
        end_score=end_score,
        max_steps=max_steps,
        render_mode="human",
    )
    model = ConvActorCritic(
        input_channels=env.observation_space.shape[0],
        output_dim=env.action_space.n,
        grid_size=grid_size,
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
        entropy_coef=float(config.get("entropy_coef", 0.01)),
    ).to(device)
    model.load_checkpoint(model_path)
    model.eval()

    observation, _ = env.reset(seed=seed)
    try:
        while not env.closed:
            action = model.predict(as_tensor(observation, device), deterministic=True)
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
    parser = argparse.ArgumentParser(description="Play a trained Snake checkpoint")
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=Path("runs/snake/checkpoints/best.pt"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    arguments = parser.parse_args()
    play_game(
        arguments.checkpoint,
        device_name=arguments.device,
        seed=arguments.seed,
        fps=arguments.fps,
    )


if __name__ == "__main__":
    main()

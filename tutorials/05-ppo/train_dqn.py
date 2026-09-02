"""Train a low-compute DQN agent in the Snake environment."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dqn import DQNAgent, ReplayBuffer
from environment import SnakeEnv
from main import choose_device, seed_everything
from torch.utils.tensorboard import SummaryWriter


@dataclass(slots=True)
class DQNTrainConfig:
    episodes: int = 1_000
    grid_size: int = 6
    end_score: int = 4
    max_steps: int = 100
    replay_capacity: int = 20_000
    batch_size: int = 64
    learning_starts: int = 250
    train_interval: int = 4
    target_update_interval: int = 250
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8_000
    eval_interval: int = 200
    eval_episodes: int = 20
    save_interval: int = 500
    seed: int = 42
    torch_threads: int = 1
    device: str = "auto"
    output_dir: Path = Path("runs/dqn")
    tensorboard: bool = True

    def validate(self) -> None:
        positive = {
            "episodes": self.episodes,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "train_interval": self.train_interval,
            "target_update_interval": self.target_update_interval,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "eval_episodes": self.eval_episodes,
            "torch_threads": self.torch_threads,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.eval_interval < 0 or self.save_interval < 0:
            raise ValueError("intervals cannot be negative")

    def checkpoint_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def epsilon_at_step(config: DQNTrainConfig, total_steps: int) -> float:
    fraction = min(max(total_steps, 0) / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def evaluate_dqn(
    agent: DQNAgent,
    config: DQNTrainConfig,
    *,
    seed_offset: int,
) -> dict[str, float]:
    env = SnakeEnv(
        grid_size=config.grid_size,
        end_score=config.end_score,
        max_steps=config.max_steps,
    )
    random_source = random.Random(config.seed + seed_offset)
    rewards: list[float] = []
    scores: list[int] = []
    successes = 0
    agent.online.eval()
    try:
        for episode in range(config.eval_episodes):
            observation, _ = env.reset(seed=config.seed + seed_offset + episode)
            episode_reward = 0.0
            while True:
                action = agent.select_action(
                    observation,
                    epsilon=0.0,
                    random_source=random_source,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    rewards.append(episode_reward)
                    scores.append(int(info["score"]))
                    successes += int(info["length"] >= config.end_score)
                    break
    finally:
        env.close()
        agent.online.train()
    return {
        "reward": float(np.mean(rewards)),
        "score": float(np.mean(scores)),
        "success_rate": successes / config.eval_episodes,
    }


def train_dqn(config: DQNTrainConfig) -> dict[str, Any]:
    config.validate()
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)
    device = choose_device(config.device)
    random_source = random.Random(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    env = SnakeEnv(
        grid_size=config.grid_size,
        end_score=config.end_score,
        max_steps=config.max_steps,
    )
    agent = DQNAgent(
        input_channels=env.observation_space.shape[0],
        action_count=env.action_space.n,
        grid_size=config.grid_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )
    replay = ReplayBuffer(config.replay_capacity, seed=config.seed)
    writer = SummaryWriter(config.output_dir / "tensorboard") if config.tensorboard else None
    history: dict[str, list[Any]] = {
        "train_reward": [],
        "train_score": [],
        "epsilon": [],
        "eval": [],
    }
    best_eval_reward = -float("inf")
    total_steps = 0
    try:
        for episode in range(config.episodes):
            observation, _ = env.reset(seed=config.seed + episode)
            episode_reward = 0.0
            losses: list[float] = []
            while True:
                epsilon = epsilon_at_step(config, total_steps)
                action = agent.select_action(
                    observation,
                    epsilon=epsilon,
                    random_source=random_source,
                )
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                replay.append(observation, action, reward, next_observation, done)
                observation = next_observation
                episode_reward += reward
                total_steps += 1

                ready = total_steps >= config.learning_starts and len(replay) >= config.batch_size
                if ready and total_steps % config.train_interval == 0:
                    metrics = agent.update(replay.sample(config.batch_size, device), config.gamma)
                    losses.append(metrics["loss"])
                    if writer is not None:
                        for name, value in metrics.items():
                            writer.add_scalar(f"train/{name}", value, total_steps)
                if total_steps % config.target_update_interval == 0:
                    agent.sync_target()
                if done:
                    score = int(info["score"])
                    break

            history["train_reward"].append(episode_reward)
            history["train_score"].append(score)
            history["epsilon"].append(epsilon_at_step(config, total_steps))
            if writer is not None:
                writer.add_scalar("train/reward", episode_reward, episode)
                writer.add_scalar("train/score", score, episode)
                writer.add_scalar("train/epsilon", history["epsilon"][-1], episode)
                if losses:
                    writer.add_scalar("train/episode_loss", float(np.mean(losses)), episode)

            if episode == 0 or (episode + 1) % 25 == 0:
                print(
                    f"episode={episode + 1}/{config.episodes} reward={episode_reward:.2f} "
                    f"score={score} epsilon={history['epsilon'][-1]:.3f}"
                )

            if config.eval_interval and (episode + 1) % config.eval_interval == 0:
                evaluation = evaluate_dqn(
                    agent,
                    config,
                    seed_offset=100_000 + episode * config.eval_episodes,
                )
                evaluation["episode"] = episode + 1
                history["eval"].append(evaluation)
                print(
                    f"evaluation episode={episode + 1} reward={evaluation['reward']:.2f} "
                    f"score={evaluation['score']:.2f} "
                    f"success={evaluation['success_rate']:.0%}"
                )
                if writer is not None:
                    for name, value in evaluation.items():
                        if name != "episode":
                            writer.add_scalar(f"eval/{name}", value, episode)
                if evaluation["reward"] > best_eval_reward:
                    best_eval_reward = evaluation["reward"]
                    agent.save_checkpoint(
                        checkpoint_dir / "best.pt",
                        episode=episode,
                        total_steps=total_steps,
                        config=config.checkpoint_payload(),
                    )

            if config.save_interval and (episode + 1) % config.save_interval == 0:
                agent.save_checkpoint(
                    checkpoint_dir / f"episode-{episode + 1}.pt",
                    episode=episode,
                    total_steps=total_steps,
                    config=config.checkpoint_payload(),
                )
    finally:
        env.close()
        if writer is not None:
            writer.close()

    latest_path = checkpoint_dir / "latest.pt"
    agent.save_checkpoint(
        latest_path,
        episode=config.episodes - 1,
        total_steps=total_steps,
        config=config.checkpoint_payload(),
    )
    summary: dict[str, Any] = {
        "algorithm": "dqn",
        "episodes": config.episodes,
        "total_steps": total_steps,
        "last_reward": history["train_reward"][-1],
        "last_score": history["train_score"][-1],
        "best_eval_reward": best_eval_reward if np.isfinite(best_eval_reward) else None,
        "device": str(device),
        "checkpoint": str(latest_path),
    }
    (config.output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Snake DQN lesson")
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--end-score", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=250)
    parser.add_argument("--train-interval", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=8_000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dqn"))
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = DQNTrainConfig(
        episodes=arguments.episodes,
        grid_size=arguments.grid_size,
        end_score=arguments.end_score,
        max_steps=arguments.max_steps,
        replay_capacity=arguments.replay_capacity,
        batch_size=arguments.batch_size,
        learning_starts=arguments.learning_starts,
        train_interval=arguments.train_interval,
        target_update_interval=arguments.target_update_interval,
        learning_rate=arguments.learning_rate,
        gamma=arguments.gamma,
        epsilon_start=arguments.epsilon_start,
        epsilon_end=arguments.epsilon_end,
        epsilon_decay_steps=arguments.epsilon_decay_steps,
        eval_interval=arguments.eval_interval,
        eval_episodes=arguments.eval_episodes,
        save_interval=arguments.save_interval,
        seed=arguments.seed,
        torch_threads=arguments.torch_threads,
        device=arguments.device,
        output_dir=arguments.output_dir,
        tensorboard=not arguments.no_tensorboard,
    )
    print(json.dumps(train_dqn(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

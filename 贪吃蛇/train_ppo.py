"""Train a low-compute PPO agent in the Snake environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from environment import SnakeEnv
from main import choose_device, seed_everything
from ppo import PPOAgent, RolloutBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass(slots=True)
class PPOTrainConfig:
    episodes: int = 1_000
    grid_size: int = 6
    end_score: int = 4
    max_steps: int = 100
    rollout_steps: int = 256
    update_epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 0.1
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    eval_interval: int = 200
    eval_episodes: int = 20
    save_interval: int = 500
    seed: int = 42
    torch_threads: int = 1
    device: str = "auto"
    output_dir: Path = Path("runs/ppo")
    tensorboard: bool = True

    def validate(self) -> None:
        positive = {
            "episodes": self.episodes,
            "rollout_steps": self.rollout_steps,
            "update_epochs": self.update_epochs,
            "batch_size": self.batch_size,
            "eval_episodes": self.eval_episodes,
            "torch_threads": self.torch_threads,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        unit_interval = {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
        }
        for name, value in unit_interval.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0 or self.reward_scale <= 0.0:
            raise ValueError("learning_rate, reward_scale, and max_grad_norm must be positive")
        if self.value_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError("loss coefficients cannot be negative")
        if self.eval_interval < 0 or self.save_interval < 0:
            raise ValueError("intervals cannot be negative")

    def checkpoint_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def evaluate_ppo(
    agent: PPOAgent,
    config: PPOTrainConfig,
    *,
    seed_offset: int,
) -> dict[str, float]:
    env = SnakeEnv(
        grid_size=config.grid_size,
        end_score=config.end_score,
        max_steps=config.max_steps,
    )
    rewards: list[float] = []
    scores: list[int] = []
    successes = 0
    generator = torch.Generator(device=agent.device)
    generator.manual_seed(config.seed + seed_offset)
    was_training = agent.model.training
    agent.model.eval()
    try:
        for episode in range(config.eval_episodes):
            observation, _ = env.reset(seed=config.seed + seed_offset + episode)
            episode_reward = 0.0
            while True:
                action = agent.predict(observation, generator=generator)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    rewards.append(episode_reward)
                    scores.append(int(info["score"]))
                    successes += int(info["length"] >= config.end_score)
                    break
    finally:
        env.close()
        agent.model.train(was_training)
    return {
        "reward": float(np.mean(rewards)),
        "score": float(np.mean(scores)),
        "success_rate": successes / config.eval_episodes,
    }


def train_ppo(config: PPOTrainConfig) -> dict[str, Any]:
    config.validate()
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)
    device = choose_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    env = SnakeEnv(
        grid_size=config.grid_size,
        end_score=config.end_score,
        max_steps=config.max_steps,
    )
    agent = PPOAgent(
        input_channels=env.observation_space.shape[0],
        action_count=env.action_space.n,
        grid_size=config.grid_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )
    rollout = RolloutBuffer()
    writer = SummaryWriter(config.output_dir / "tensorboard") if config.tensorboard else None
    history: dict[str, list[Any]] = {
        "train_reward": [],
        "train_score": [],
        "updates": [],
        "eval": [],
    }
    best_eval_reward = -float("inf")
    total_steps = 0
    try:
        for episode in range(config.episodes):
            observation, _ = env.reset(seed=config.seed + episode)
            episode_reward = 0.0
            while True:
                action, log_probability, value = agent.select_action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rollout.append(
                    observation,
                    action,
                    reward * config.reward_scale,
                    done,
                    log_probability,
                    value,
                )
                observation = next_observation
                episode_reward += reward
                total_steps += 1
                if done:
                    score = int(info["score"])
                    break

            history["train_reward"].append(episode_reward)
            history["train_score"].append(score)
            if writer is not None:
                writer.add_scalar("train/reward", episode_reward, episode)
                writer.add_scalar("train/score", score, episode)

            should_update = len(rollout) >= config.rollout_steps or episode + 1 == config.episodes
            if should_update:
                metrics = agent.update(
                    rollout.build_batch(
                        device,
                        gamma=config.gamma,
                        gae_lambda=config.gae_lambda,
                    ),
                    update_epochs=config.update_epochs,
                    batch_size=config.batch_size,
                    clip_ratio=config.clip_ratio,
                    value_coef=config.value_coef,
                    entropy_coef=config.entropy_coef,
                    max_grad_norm=config.max_grad_norm,
                )
                metrics["episode"] = episode + 1
                metrics["samples"] = len(rollout)
                history["updates"].append(metrics)
                if writer is not None:
                    for name, metric in metrics.items():
                        if name not in {"episode", "samples"}:
                            writer.add_scalar(f"ppo/{name}", metric, total_steps)
                rollout.clear()

            if episode == 0 or (episode + 1) % 25 == 0:
                print(
                    f"episode={episode + 1}/{config.episodes} reward={episode_reward:.2f} "
                    f"score={score} collected={len(rollout)}/{config.rollout_steps}"
                )

            if config.eval_interval and (episode + 1) % config.eval_interval == 0:
                evaluation = evaluate_ppo(
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
                    for name, metric in evaluation.items():
                        if name != "episode":
                            writer.add_scalar(f"eval/{name}", metric, episode)
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
        "algorithm": "ppo",
        "episodes": config.episodes,
        "total_steps": total_steps,
        "updates": len(history["updates"]),
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
    parser = argparse.ArgumentParser(description="Train the Snake PPO lesson")
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--end-score", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = PPOTrainConfig(
        episodes=arguments.episodes,
        grid_size=arguments.grid_size,
        end_score=arguments.end_score,
        max_steps=arguments.max_steps,
        rollout_steps=arguments.rollout_steps,
        update_epochs=arguments.update_epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gamma=arguments.gamma,
        gae_lambda=arguments.gae_lambda,
        reward_scale=arguments.reward_scale,
        clip_ratio=arguments.clip_ratio,
        value_coef=arguments.value_coef,
        entropy_coef=arguments.entropy_coef,
        max_grad_norm=arguments.max_grad_norm,
        eval_interval=arguments.eval_interval,
        eval_episodes=arguments.eval_episodes,
        save_interval=arguments.save_interval,
        seed=arguments.seed,
        torch_threads=arguments.torch_threads,
        device=arguments.device,
        output_dir=arguments.output_dir,
        tensorboard=not arguments.no_tensorboard,
    )
    print(json.dumps(train_ppo(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

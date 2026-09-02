"""Train the low-compute Snake Actor-Critic agent."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from environment import SnakeEnv
from model import ConvActorCritic
from torch.utils.tensorboard import SummaryWriter


@dataclass(slots=True)
class TrainConfig:
    episodes: int = 500
    teacher_episodes: int = 100
    imitation_updates: int = 4
    grid_size: int = 6
    end_score: int = 4
    max_steps: int = 100
    eval_interval: int = 100
    eval_episodes: int = 20
    save_interval: int = 250
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gamma: float = 0.99
    entropy_coef: float = 0.01
    seed: int = 42
    device: str = "auto"
    output_dir: Path = Path("runs/snake")
    resume: Path | None = None
    tensorboard: bool = True

    def validate(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if not 0 <= self.teacher_episodes <= self.episodes:
            raise ValueError("teacher_episodes must be between zero and episodes")
        if self.imitation_updates < 1:
            raise ValueError("imitation_updates must be positive")
        if self.eval_interval < 0 or self.save_interval < 0:
            raise ValueError("intervals cannot be negative")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be positive")

    def checkpoint_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["resume"] = str(self.resume) if self.resume else None
        return payload


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def as_tensor(observation: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(observation).unsqueeze(0).to(device=device, dtype=torch.float32)


def evaluate_model(
    agent: ConvActorCritic,
    config: TrainConfig,
    device: torch.device,
    *,
    seed_offset: int,
) -> dict[str, float]:
    """Evaluate without teacher actions in an independent environment."""

    env = SnakeEnv(
        grid_size=config.grid_size,
        end_score=config.end_score,
        max_steps=config.max_steps,
    )
    rewards: list[float] = []
    scores: list[int] = []
    successes = 0
    was_training = agent.training
    agent.eval()
    try:
        for episode in range(config.eval_episodes):
            observation, _ = env.reset(seed=config.seed + seed_offset + episode)
            episode_reward = 0.0
            while True:
                action = agent.predict(as_tensor(observation, device), deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    rewards.append(episode_reward)
                    scores.append(int(info["score"]))
                    if info["length"] >= config.end_score:
                        successes += 1
                    break
    finally:
        env.close()
        agent.train(was_training)

    return {
        "reward": float(np.mean(rewards)),
        "score": float(np.mean(scores)),
        "success_rate": successes / config.eval_episodes,
    }


def train(config: TrainConfig) -> dict[str, float | int | str | None]:
    config.validate()
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
    agent = ConvActorCritic(
        input_channels=env.observation_space.shape[0],
        output_dim=env.action_space.n,
        grid_size=config.grid_size,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        entropy_coef=config.entropy_coef,
    ).to(device)

    start_episode = 0
    if config.resume is not None:
        metadata = agent.load_checkpoint(config.resume)
        start_episode = metadata["episode"] + 1

    writer = SummaryWriter(config.output_dir / "tensorboard") if config.tensorboard else None
    best_eval_reward = -float("inf")
    last_reward = 0.0
    last_score = 0
    demonstrations: deque[tuple[np.ndarray, int]] = deque(maxlen=10_000)
    try:
        for episode in range(start_episode, config.episodes):
            observation, _ = env.reset(seed=config.seed + episode)
            episode_reward = 0.0
            imitation_losses: list[float] = []
            value_losses: list[float] = []
            update_metrics: dict[str, float] = {}
            teacher_phase = episode < config.teacher_episodes

            while True:
                state = as_tensor(observation, device)
                action = env.teacher_action() if teacher_phase else agent.predict(state)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if teacher_phase:
                    demonstrations.append((observation.copy(), action))
                    demonstration_list = list(demonstrations)
                    for _ in range(config.imitation_updates):
                        batch = random.sample(
                            demonstration_list,
                            k=min(64, len(demonstration_list)),
                        )
                        batch_states = torch.from_numpy(
                            np.stack([sample[0] for sample in batch])
                        ).to(device=device, dtype=torch.float32)
                        batch_actions = torch.tensor(
                            [sample[1] for sample in batch],
                            device=device,
                        )
                        imitation_losses.append(agent.imitation_update(batch_states, batch_actions))
                    value_losses.append(
                        agent.value_update(
                            state,
                            reward,
                            as_tensor(next_observation, device),
                            done,
                            config.gamma,
                        )
                    )
                else:
                    update_metrics = agent.update(
                        state,
                        action,
                        reward,
                        as_tensor(next_observation, device),
                        done,
                        config.gamma,
                    )

                episode_reward += reward
                observation = next_observation
                if done:
                    last_score = int(info["score"])
                    break

            last_reward = episode_reward
            phase = "teacher" if teacher_phase else "actor-critic"
            if writer is not None:
                writer.add_scalar("train/reward", episode_reward, episode)
                writer.add_scalar("train/score", last_score, episode)
                if imitation_losses:
                    writer.add_scalar(
                        "train/imitation_loss",
                        float(np.mean(imitation_losses)),
                        episode,
                    )
                if value_losses:
                    writer.add_scalar(
                        "train/teacher_value_loss",
                        float(np.mean(value_losses)),
                        episode,
                    )
                for name, value in update_metrics.items():
                    writer.add_scalar(f"train/{name}", value, episode)

            if episode == start_episode or (episode + 1) % 10 == 0:
                print(
                    f"episode={episode + 1}/{config.episodes} phase={phase} "
                    f"reward={episode_reward:.2f} score={last_score}"
                )

            if config.eval_interval and (episode + 1) % config.eval_interval == 0:
                evaluation = evaluate_model(
                    agent,
                    config,
                    device,
                    seed_offset=100_000 + episode * config.eval_episodes,
                )
                print(
                    f"evaluation episode={episode + 1} reward={evaluation['reward']:.2f} "
                    f"score={evaluation['score']:.2f} "
                    f"success={evaluation['success_rate']:.0%}"
                )
                if writer is not None:
                    for name, value in evaluation.items():
                        writer.add_scalar(f"eval/{name}", value, episode)
                if evaluation["reward"] > best_eval_reward:
                    best_eval_reward = evaluation["reward"]
                    agent.save_checkpoint(
                        checkpoint_dir / "best.pt",
                        episode=episode,
                        config=config.checkpoint_payload(),
                    )

            if config.save_interval and (episode + 1) % config.save_interval == 0:
                agent.save_checkpoint(
                    checkpoint_dir / f"episode-{episode + 1}.pt",
                    episode=episode,
                    config=config.checkpoint_payload(),
                )
    finally:
        env.close()
        if writer is not None:
            writer.close()

    final_episode = max(start_episode, config.episodes) - 1
    final_path = checkpoint_dir / "latest.pt"
    agent.save_checkpoint(
        final_path,
        episode=final_episode,
        config=config.checkpoint_payload(),
    )
    summary: dict[str, float | int | str | None] = {
        "episodes": config.episodes,
        "last_reward": last_reward,
        "last_score": last_score,
        "best_eval_reward": (best_eval_reward if np.isfinite(best_eval_reward) else None),
        "device": str(device),
        "checkpoint": str(final_path),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Snake Actor-Critic lesson")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--teacher-episodes", type=int, default=100)
    parser.add_argument("--imitation-updates", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--end-score", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/snake"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = TrainConfig(
        episodes=arguments.episodes,
        teacher_episodes=arguments.teacher_episodes,
        imitation_updates=arguments.imitation_updates,
        grid_size=arguments.grid_size,
        end_score=arguments.end_score,
        max_steps=arguments.max_steps,
        eval_interval=arguments.eval_interval,
        eval_episodes=arguments.eval_episodes,
        save_interval=arguments.save_interval,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gamma=arguments.gamma,
        entropy_coef=arguments.entropy_coef,
        seed=arguments.seed,
        device=arguments.device,
        output_dir=arguments.output_dir,
        resume=arguments.resume,
        tensorboard=not arguments.no_tensorboard,
    )
    summary = train(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

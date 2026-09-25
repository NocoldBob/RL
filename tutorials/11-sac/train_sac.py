"""Train Soft Actor-Critic on MountainCarContinuous-v0."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from mountain_car_baselines import ENV_ID
from sac import ReplayBuffer, SACAgent, normalize_observation


@dataclass(slots=True)
class SACConfig:
    total_steps: int = 50_000
    replay_capacity: int = 100_000
    learning_starts: int = 5_000
    random_action_repeat: int = 10
    batch_size: int = 128
    updates_per_step: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 0.01
    initial_alpha: float = 0.2
    eval_interval: int = 10_000
    eval_episodes: int = 20
    seed: int = 42
    torch_threads: int = 1
    output_dir: Path = Path("runs/sac/seed-42")

    def validate(self) -> None:
        for name in (
            "total_steps",
            "replay_capacity",
            "learning_starts",
            "random_action_repeat",
            "batch_size",
            "updates_per_step",
            "eval_episodes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.learning_rate <= 0.0 or self.reward_scale <= 0.0:
            raise ValueError("learning_rate and reward_scale must be positive")

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_sac_agent(
    agent: SACAgent,
    seeds: list[int],
    *,
    capture_seed: int | None = None,
) -> tuple[dict[str, float | int | None], list[dict[str, Any]], dict[str, list[float]] | None]:
    """Evaluate deterministic actions on fixed initial states."""
    env = gym.make(ENV_ID)
    rows: list[dict[str, Any]] = []
    captured_trace = None
    was_training = agent.actor.training
    agent.actor.eval()
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            episode_return = 0.0
            action_energy = 0.0
            positions = [float(observation[0])]
            velocities = [float(observation[1])]
            actions: list[float] = []
            rewards: list[float] = []
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                action = agent.predict(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                action_value = float(action[0])
                episode_return += float(reward)
                action_energy += action_value**2
                steps += 1
                if seed == capture_seed:
                    positions.append(float(observation[0]))
                    velocities.append(float(observation[1]))
                    actions.append(action_value)
                    rewards.append(float(reward))
            rows.append(
                {
                    "seed": seed,
                    "episode_return": episode_return,
                    "success": bool(terminated),
                    "steps": steps,
                    "action_energy": action_energy,
                    "final_position": float(observation[0]),
                }
            )
            if seed == capture_seed:
                captured_trace = {
                    "position": positions,
                    "velocity": velocities,
                    "action": actions,
                    "reward": rewards,
                }
    finally:
        env.close()
        agent.actor.train(was_training)

    success_steps = [row["steps"] for row in rows if row["success"]]
    summary: dict[str, float | int | None] = {
        "episodes": len(rows),
        "average_return": float(np.mean([row["episode_return"] for row in rows])),
        "return_std": float(np.std([row["episode_return"] for row in rows])),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "average_steps": float(np.mean([row["steps"] for row in rows])),
        "average_success_steps": float(np.mean(success_steps)) if success_steps else None,
        "average_action_energy": float(np.mean([row["action_energy"] for row in rows])),
        "average_action_cost": float(0.1 * np.mean([row["action_energy"] for row in rows])),
    }
    return summary, rows, captured_trace


def train_sac(config: SACConfig) -> dict[str, Any]:
    config.validate()
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = config.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    env = gym.make(ENV_ID)
    agent = SACAgent(
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        initial_alpha=config.initial_alpha,
    )
    replay = ReplayBuffer(config.replay_capacity, 2, 1, seed=config.seed)
    exploration_rng = np.random.default_rng(config.seed)
    exploration_action = exploration_rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
    history: dict[str, list[Any]] = {"episodes": [], "updates": [], "evaluations": []}
    observation, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_steps = 0
    episode_count = 0
    best_return = -float("inf")

    try:
        for step in range(1, config.total_steps + 1):
            if step <= config.learning_starts:
                if (step - 1) % config.random_action_repeat == 0:
                    exploration_action = exploration_rng.uniform(-1.0, 1.0, size=1).astype(
                        np.float32
                    )
                action = exploration_action
            else:
                action = agent.predict(observation, deterministic=False)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.append(
                normalize_observation(observation),
                action,
                float(reward) * config.reward_scale,
                normalize_observation(next_observation),
                terminated,
            )
            episode_return += float(reward)
            episode_steps += 1
            observation = next_observation

            if step > config.learning_starts and len(replay) >= config.batch_size:
                metrics = {}
                for _ in range(config.updates_per_step):
                    metrics = agent.update(replay.sample(config.batch_size, agent.device))
                if step % 100 == 0:
                    history["updates"].append({"step": step, **metrics})

            if done:
                history["episodes"].append(
                    {
                        "step": step,
                        "return": episode_return,
                        "length": episode_steps,
                        "success": bool(terminated),
                    }
                )
                episode_count += 1
                observation, _ = env.reset()
                episode_return = 0.0
                episode_steps = 0

            should_evaluate = config.eval_interval and (
                step % config.eval_interval == 0 or step == config.total_steps
            )
            if should_evaluate:
                evaluation, _, _ = evaluate_sac_agent(
                    agent, list(range(50_000, 50_000 + config.eval_episodes))
                )
                evaluation["step"] = step
                history["evaluations"].append(evaluation)
                print(
                    f"seed={config.seed} step={step}/{config.total_steps} "
                    f"episodes={episode_count} return={evaluation['average_return']:.2f} "
                    f"success={evaluation['success_rate']:.0%} "
                    f"alpha={float(agent.alpha.detach()):.3f}"
                )
                if float(evaluation["average_return"]) > best_return:
                    best_return = float(evaluation["average_return"])
                    agent.save_checkpoint(
                        checkpoints / "best.pt", steps=step, config=config.payload()
                    )
    finally:
        env.close()

    latest = checkpoints / "latest.pt"
    agent.save_checkpoint(latest, steps=config.total_steps, config=config.payload())
    summary = {
        "algorithm": "sac",
        "seed": config.seed,
        "total_steps": config.total_steps,
        "episodes": episode_count,
        "replay_size": len(replay),
        "best_eval_return": best_return,
        "final_alpha": float(agent.alpha.detach()),
        "checkpoint": str(latest),
    }
    (config.output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAC on continuous Mountain Car")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--random-action-repeat", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or Path(f"runs/sac/seed-{args.seed}")
    config = SACConfig(
        total_steps=args.total_steps,
        learning_starts=args.learning_starts,
        random_action_repeat=args.random_action_repeat,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        output_dir=output_dir,
    )
    print(json.dumps(train_sac(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

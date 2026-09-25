"""Train continuous PPO on LunarLanderContinuous-v3."""

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
from lunar_agents import LunarPPOAgent, normalize_lunar_observation
from lunar_baselines import ENV_ID
from lunar_evaluation import evaluate_lunar_agent
from lunar_ppo_components import ContinuousRolloutBuffer


@dataclass(slots=True)
class LunarPPOConfig:
    total_steps: int = 300_000
    rollout_steps: int = 2048
    update_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 0.01
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    eval_interval: int = 50_000
    eval_episodes: int = 20
    seed: int = 42
    torch_threads: int = 1
    output_dir: Path = Path("runs/lunar-ppo/seed-42")

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result


def train_lunar_ppo(config: LunarPPOConfig) -> dict[str, Any]:
    torch.set_num_threads(config.torch_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = config.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    env = gym.make(ENV_ID)
    agent = LunarPPOAgent(learning_rate=config.learning_rate)
    rollout = ContinuousRolloutBuffer()
    history: dict[str, list[Any]] = {"episodes": [], "updates": [], "evaluations": []}
    observation, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_steps = 0
    best_return = -float("inf")
    try:
        for step in range(1, config.total_steps + 1):
            action, log_probability, value = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_value = 0.0 if terminated else agent.value(next_observation)
            rollout.append(
                normalize_lunar_observation(observation),
                action,
                float(reward) * config.reward_scale,
                terminated,
                done,
                log_probability,
                value,
                next_value,
            )
            episode_return += float(reward)
            episode_steps += 1
            observation = next_observation
            if done:
                history["episodes"].append(
                    {
                        "step": step,
                        "return": episode_return,
                        "length": episode_steps,
                        "landed": bool(terminated and reward == 100.0),
                    }
                )
                observation, _ = env.reset()
                episode_return = 0.0
                episode_steps = 0
            if len(rollout) >= config.rollout_steps or step == config.total_steps:
                learning_rate = config.learning_rate * max(0.0, 1.0 - step / config.total_steps)
                for group in agent.optimizer.param_groups:
                    group["lr"] = learning_rate
                metrics = agent.update(
                    rollout.build_batch(
                        agent.device, gamma=config.gamma, gae_lambda=config.gae_lambda
                    ),
                    update_epochs=config.update_epochs,
                    batch_size=config.batch_size,
                    clip_ratio=config.clip_ratio,
                    value_coef=config.value_coef,
                    entropy_coef=config.entropy_coef,
                    max_grad_norm=config.max_grad_norm,
                )
                history["updates"].append({"step": step, "samples": len(rollout), **metrics})
                rollout.clear()
            if step % config.eval_interval == 0 or step == config.total_steps:
                summary, _, _ = evaluate_lunar_agent(
                    agent, list(range(50_000, 50_000 + config.eval_episodes))
                )
                history["evaluations"].append({"step": step, **summary})
                (config.output_dir / "history.json").write_text(
                    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    f"PPO seed={config.seed} step={step} "
                    f"return={summary['average_return']:.1f} "
                    f"landing={summary['landing_rate']:.0%}"
                )
                if summary["average_return"] > best_return:
                    best_return = summary["average_return"]
                    agent.save_checkpoint(
                        checkpoints / "best.pt", steps=step, config=config.payload()
                    )
    finally:
        env.close()
    agent.save_checkpoint(
        checkpoints / "latest.pt", steps=config.total_steps, config=config.payload()
    )
    summary = {
        "algorithm": "lunar_ppo",
        "seed": config.seed,
        "total_steps": config.total_steps,
        "best_eval_return": best_return,
    }
    (config.output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on LunarLanderContinuous-v3")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(f"runs/lunar-ppo/seed-{args.seed}")
    print(
        json.dumps(
            train_lunar_ppo(
                LunarPPOConfig(
                    total_steps=args.total_steps,
                    seed=args.seed,
                    eval_interval=args.eval_interval,
                    eval_episodes=args.eval_episodes,
                    output_dir=output,
                )
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

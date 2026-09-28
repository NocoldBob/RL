"""Train the online behavior policy used to produce offline datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
from acrobot_env import evaluate_policy
from online_dqn import DQNAgent, OnlineConfig, ReplayBuffer, seed_everything


def train(config: OnlineConfig) -> dict[str, float]:
    torch.set_num_threads(1)
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    environment = gym.make("Acrobot-v1")
    agent = DQNAgent(config.learning_rate)
    replay = ReplayBuffer()
    observation, _ = environment.reset(seed=config.seed)
    history = {"episodes": [], "updates": [], "evaluations": []}
    episode_return = 0.0
    episode_steps = 0
    best_return = -float("inf")
    try:
        for step in range(1, config.total_steps + 1):
            fraction = min(1.0, step / config.epsilon_decay_steps)
            epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)
            action = agent.act(observation, epsilon)
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            replay.append(observation, action, reward, next_observation, terminated)
            observation = next_observation
            episode_return += float(reward)
            episode_steps += 1
            if terminated or truncated:
                history["episodes"].append(
                    {"step": step, "return": episode_return, "length": episode_steps}
                )
                observation, _ = environment.reset()
                episode_return = 0.0
                episode_steps = 0
            if len(replay) >= max(config.warmup_steps, config.batch_size):
                metrics = agent.update(replay.sample(config.batch_size, agent.device), config.gamma)
                if step % 100 == 0:
                    history["updates"].append({"step": step, **metrics})
            if step % config.target_interval == 0:
                agent.sync_target()
            if step % config.eval_interval == 0 or step == config.total_steps:
                report = evaluate_policy(
                    agent.act,
                    list(range(50_000, 50_000 + config.eval_episodes)),
                )["summary"]
                history["evaluations"].append({"step": step, "epsilon": epsilon, **report})
                (config.output_dir / "history.json").write_text(
                    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"online DQN step={step} return={report['average_return']:.1f}")
                agent.save(
                    config.output_dir / "checkpoints" / f"step-{step}.pt", step, config.payload()
                )
                if report["average_return"] > best_return:
                    best_return = report["average_return"]
                    agent.save(
                        config.output_dir / "checkpoints" / "best.pt", step, config.payload()
                    )
    finally:
        environment.close()
    return {"best_average_return": best_return, "total_steps": float(config.total_steps)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/acrobot-online-dqn/seed-42"))
    args = parser.parse_args()
    result = train(
        OnlineConfig(total_steps=args.total_steps, seed=args.seed, output_dir=args.output_dir)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

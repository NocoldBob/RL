"""Train CNN Double DQN with a deliberately small discrete driving action set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from carracing_dqn import DQNConfig, VisualDQNAgent, VisualReplayBuffer, seed_everything
from carracing_env import DrivingConfig, VisualCarRacing, run_episode, summarize_episodes


def evaluate(agent: VisualDQNAgent, config: DQNConfig) -> dict[str, float]:
    env = VisualCarRacing(
        DrivingConfig(frame_stack=config.frame_stack, action_mode="discrete", max_decisions=250)
    )
    try:
        rows = [
            run_episode(env, lambda obs: agent.act(obs), seed=50_000 + i)
            for i in range(config.eval_episodes)
        ]
        return summarize_episodes(rows)
    finally:
        env.close()


def train(config: DQNConfig) -> dict[str, Any]:
    torch.set_num_threads(1)
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    env = VisualCarRacing(
        DrivingConfig(
            frame_stack=config.frame_stack,
            action_mode="discrete",
            max_decisions=250,
            seed=config.seed,
        )
    )
    agent = VisualDQNAgent(
        env.observation_shape[0],
        env.action_count,
        image_size=config.image_size,
        learning_rate=config.learning_rate,
    )
    replay = VisualReplayBuffer(config.replay_capacity, env.observation_shape)
    observation, _ = env.reset(seed=config.seed)
    history: dict[str, list[dict[str, Any]]] = {"episodes": [], "updates": [], "evaluations": []}
    episode_return = 0.0
    episode_length = 0
    best_return = -float("inf")
    try:
        for step in range(1, config.total_steps + 1):
            fraction = min(1.0, step / config.epsilon_decay_steps)
            epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)
            action = agent.act(observation, epsilon)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.append(observation, action, reward, next_observation, done)
            observation = next_observation
            episode_return += float(info["raw_reward"])
            episode_length += 1
            if done:
                history["episodes"].append(
                    {"step": step, "return": episode_return, "length": episode_length}
                )
                observation, _ = env.reset()
                episode_return = 0.0
                episode_length = 0
            if (
                len(replay) >= max(config.warmup_steps, config.batch_size)
                and step % config.train_interval == 0
            ):
                metrics = agent.update(replay.sample(config.batch_size, agent.device), config.gamma)
                if step % 100 == 0:
                    history["updates"].append({"step": step, **metrics})
            if step % config.target_interval == 0:
                agent.sync_target()
            if step % config.eval_interval == 0 or step == config.total_steps:
                metrics = evaluate(agent, config)
                history["evaluations"].append({"step": step, "epsilon": epsilon, **metrics})
                (config.output_dir / "history.json").write_text(
                    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"DQN step={step} return={metrics['average_return']:.1f}")
                if metrics["average_return"] > best_return:
                    best_return = metrics["average_return"]
                    agent.save(
                        config.output_dir / "checkpoints" / "best.pt",
                        steps=step,
                        config=config.payload(),
                    )
    finally:
        env.close()
    agent.save(
        config.output_dir / "checkpoints" / "latest.pt",
        steps=config.total_steps,
        config=config.payload(),
    )
    summary = {
        "algorithm": "cnn_double_dqn",
        "best_average_return": best_return,
        **config.payload(),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=80_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/carracing-dqn/stack4-seed-42")
    )
    args = parser.parse_args()
    result = train(
        DQNConfig(
            frame_stack=args.frame_stack,
            total_steps=args.total_steps,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

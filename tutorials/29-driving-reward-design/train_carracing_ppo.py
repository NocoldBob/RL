"""Train continuous-action CNN PPO on CarRacing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from carracing_env import DrivingConfig, VisualCarRacing, run_episode, summarize_episodes
from carracing_ppo import PPOConfig, VisualPPOAgent, VisualRollout, seed_everything


def evaluate(
    agent: VisualPPOAgent,
    config: PPOConfig,
    *,
    corruption: str = "none",
    severity: float = 0.0,
) -> dict[str, float]:
    environment = VisualCarRacing(
        DrivingConfig(
            frame_stack=config.frame_stack,
            action_mode="continuous",
            max_decisions=250,
            corruption=corruption,
            corruption_severity=severity,
        )
    )
    try:
        rows = [
            run_episode(environment, agent.predict, seed=50_000 + index)
            for index in range(config.eval_episodes)
        ]
        return summarize_episodes(rows)
    finally:
        environment.close()


def train(config: PPOConfig) -> dict[str, Any]:
    torch.set_num_threads(1)
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    environment = VisualCarRacing(
        DrivingConfig(
            frame_stack=config.frame_stack,
            reward_mode=config.reward_mode,
            max_decisions=250,
            seed=config.seed,
        )
    )
    agent = VisualPPOAgent(
        environment.observation_shape[0],
        image_size=config.image_size,
        learning_rate=config.learning_rate,
    )
    rollout = VisualRollout()
    history: dict[str, list[dict[str, Any]]] = {"episodes": [], "updates": [], "evaluations": []}
    observation, _ = environment.reset(seed=config.seed)
    episode_return = 0.0
    episode_length = 0
    best_return = -float("inf")
    try:
        for step in range(1, config.total_steps + 1):
            action, log_probability, value = agent.act(observation)
            next_observation, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            next_value = 0.0 if terminated else agent.value(next_observation)
            rollout.observations.append(observation.copy())
            rollout.actions.append(action.copy())
            rollout.rewards.append(float(reward) * config.reward_scale)
            rollout.values.append(value)
            rollout.next_values.append(next_value)
            rollout.log_probabilities.append(log_probability)
            rollout.terminated.append(terminated)
            rollout.done.append(done)
            observation = next_observation
            episode_return += float(info["raw_reward"])
            episode_length += 1
            if done:
                history["episodes"].append(
                    {"step": step, "return": episode_return, "length": episode_length}
                )
                observation, _ = environment.reset()
                episode_return = 0.0
                episode_length = 0
            if len(rollout) >= config.rollout_steps or step == config.total_steps:
                metrics = agent.update(
                    rollout.build(agent.device, config.gamma, config.gae_lambda), config
                )
                history["updates"].append({"step": step, **metrics})
                rollout.clear()
            if step % config.eval_interval == 0 or step == config.total_steps:
                metrics = evaluate(agent, config)
                history["evaluations"].append({"step": step, **metrics})
                (config.output_dir / "history.json").write_text(
                    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"PPO step={step} return={metrics['average_return']:.1f}")
                if metrics["average_return"] > best_return:
                    best_return = metrics["average_return"]
                    agent.save(
                        config.output_dir / "checkpoints" / "best.pt",
                        steps=step,
                        config=config.payload(),
                    )
    finally:
        environment.close()
    agent.save(
        config.output_dir / "checkpoints" / "latest.pt",
        steps=config.total_steps,
        config=config.payload(),
    )
    summary = {"algorithm": "cnn_ppo", "best_average_return": best_return, **config.payload()}
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--reward-mode", choices=("raw", "shaped"), default="raw")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/carracing-ppo/raw-seed-42"))
    args = parser.parse_args()
    result = train(
        PPOConfig(
            frame_stack=args.frame_stack,
            total_steps=args.total_steps,
            reward_mode=args.reward_mode,
            augmentation=args.augmentation,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

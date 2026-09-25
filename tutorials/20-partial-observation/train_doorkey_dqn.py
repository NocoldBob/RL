"""Train full-observation DQN on DoorKey."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from doorkey_dqn import (
    DoorKeyDQNAgent,
    DQNConfig,
    GridReplayBuffer,
    door_is_open,
    evaluate_dqn,
    image_observation,
    seed_everything,
    shaped_reward,
)
from doorkey_env import make_env
from minigrid.core.world_object import Key


def train(config: DQNConfig) -> dict[str, Any]:
    torch.set_num_threads(1)
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(config.size, observation_mode=config.observation_mode)
    raw_observation, _ = env.reset(seed=config.seed)
    observation = image_observation(raw_observation)
    agent = DoorKeyDQNAgent(
        observation.shape[0],
        observation.shape[1],
        learning_rate=config.learning_rate,
        gamma=config.gamma,
    )
    replay = GridReplayBuffer(config.replay_capacity, observation.shape, seed=config.seed)
    rng = np.random.default_rng(config.seed)
    history: dict[str, list[Any]] = {"episodes": [], "updates": [], "evaluations": []}
    episode_return = 0.0
    episode_steps = 0
    best_success = -1.0
    try:
        for step in range(1, config.total_steps + 1):
            progress = min(1.0, step / config.epsilon_decay_steps)
            epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
            action = (
                int(rng.integers(env.action_space.n))
                if rng.random() < epsilon
                else agent.predict(observation)
            )
            picked_before = isinstance(env.unwrapped.carrying, Key)
            opened_before = door_is_open(env)
            raw_next_observation, raw_reward, terminated, truncated, _ = env.step(action)
            next_observation = image_observation(raw_next_observation)
            picked_after = isinstance(env.unwrapped.carrying, Key)
            opened_after = door_is_open(env)
            learning_reward = (
                shaped_reward(
                    float(raw_reward),
                    picked_before=picked_before,
                    picked_after=picked_after,
                    door_before=opened_before,
                    door_after=opened_after,
                )
                if config.reward_mode == "milestones"
                else float(raw_reward)
            )
            replay.append(observation, action, learning_reward, next_observation, terminated)
            observation = next_observation
            episode_return += float(raw_reward)
            episode_steps += 1
            if terminated or truncated:
                history["episodes"].append(
                    {
                        "step": step,
                        "return": episode_return,
                        "length": episode_steps,
                        "success": bool(terminated and episode_return > 0),
                    }
                )
                raw_observation, _ = env.reset()
                observation = image_observation(raw_observation)
                episode_return = 0.0
                episode_steps = 0
            if (
                step >= config.learning_starts
                and step % config.update_frequency == 0
                and len(replay) >= config.batch_size
            ):
                metrics = agent.update(replay.sample(config.batch_size, agent.device))
                if step % 200 == 0:
                    history["updates"].append({"step": step, "epsilon": epsilon, **metrics})
            if step % config.target_sync_interval == 0:
                agent.sync_target()
            if step % config.eval_interval == 0 or step == config.total_steps:
                summary = evaluate_dqn(
                    agent,
                    config.size,
                    list(range(50_000, 50_000 + config.eval_episodes)),
                    observation_mode=config.observation_mode,
                )
                history["evaluations"].append({"step": step, **summary})
                (config.output_dir / "history.json").write_text(
                    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    f"DQN reward={config.reward_mode} seed={config.seed} step={step} "
                    f"success={summary['success_rate']:.0%}"
                )
                if summary["success_rate"] > best_success:
                    best_success = float(summary["success_rate"])
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
        "algorithm": "doorkey_dqn",
        "reward_mode": config.reward_mode,
        "seed": config.seed,
        "total_steps": config.total_steps,
        "best_success_rate": best_success,
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full-observation DoorKey DQN")
    parser.add_argument("--reward-mode", choices=("sparse", "milestones"), default="sparse")
    parser.add_argument(
        "--observation-mode",
        choices=("compact_full", "compact_partial"),
        default="compact_full",
    )
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(
        f"runs/doorkey-dqn/{args.observation_mode}-{args.reward_mode}-seed-{args.seed}"
    )
    result = train(
        DQNConfig(
            reward_mode=args.reward_mode,
            observation_mode=args.observation_mode,
            total_steps=args.total_steps,
            seed=args.seed,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            output_dir=output,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

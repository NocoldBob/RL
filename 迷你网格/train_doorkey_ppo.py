"""Train feed-forward or GRU PPO on partial-observation DoorKey."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from doorkey_dqn import image_observation
from doorkey_env import make_env
from doorkey_ppo import DoorKeyPPOAgent, PPOConfig, RolloutBuffer, evaluate_ppo, seed_everything


def train(config: PPOConfig) -> dict[str, Any]:
    torch.set_num_threads(1)
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    def stage_size(step: int) -> int:
        if not config.curriculum:
            return config.size
        progress = step / config.total_steps
        return 5 if progress <= 0.6 else 6 if progress <= 0.8 else config.size

    current_size = stage_size(1)
    env = make_env(current_size, observation_mode="compact_partial")
    raw_observation, _ = env.reset(seed=config.seed)
    observation = image_observation(raw_observation)
    agent = DoorKeyPPOAgent(recurrent=config.recurrent, learning_rate=config.learning_rate)
    if config.initial_checkpoint is not None:
        agent.load(config.initial_checkpoint)
        if config.reference_kl_coef > 0:
            agent.set_reference_policy()
    hidden = agent.initial_hidden()
    rollout = RolloutBuffer()
    history: dict[str, list[Any]] = {"episodes": [], "updates": [], "evaluations": []}
    episode_return = 0.0
    episode_steps = 0
    best_success = -1.0
    visit_counts: dict[tuple[int, int], int] = {}
    try:
        for step in range(1, config.total_steps + 1):
            desired_size = stage_size(step)
            if desired_size != current_size:
                if len(rollout):
                    rollout.done[-1] = True
                    rollout.next_values[-1] = 0.0
                env.close()
                current_size = desired_size
                env = make_env(current_size, observation_mode="compact_partial")
                raw_observation, _ = env.reset(seed=config.seed + step)
                observation = image_observation(raw_observation)
                hidden = agent.initial_hidden()
                episode_return = 0.0
                episode_steps = 0
                visit_counts = {}
            hidden_before = hidden.detach().cpu().numpy()[0]
            action, log_probability, value, next_hidden = agent.act(observation, hidden)
            raw_next_observation, reward, terminated, truncated, _ = env.step(action)
            next_observation = image_observation(raw_next_observation)
            done = terminated or truncated
            next_value = 0.0 if terminated else agent.value(next_observation, next_hidden)
            rollout.observations.append(observation.copy())
            rollout.hidden.append(hidden_before.copy())
            rollout.actions.append(action)
            rollout.log_probabilities.append(log_probability)
            position = tuple(int(value) for value in env.unwrapped.agent_pos)
            visit_counts[position] = visit_counts.get(position, 0) + 1
            learning_reward = float(reward) + config.count_bonus / math.sqrt(visit_counts[position])
            rollout.rewards.append(learning_reward)
            rollout.values.append(value)
            rollout.next_values.append(next_value)
            rollout.terminated.append(bool(terminated))
            rollout.done.append(bool(done))
            observation = next_observation
            hidden = next_hidden.detach()
            episode_return += float(reward)
            episode_steps += 1
            if done:
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
                hidden = agent.initial_hidden()
                episode_return = 0.0
                episode_steps = 0
                visit_counts = {}
            if len(rollout) >= config.rollout_steps or step == config.total_steps:
                metrics = agent.update(
                    rollout.build(agent.device, gamma=config.gamma, gae_lambda=config.gae_lambda),
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    clip_ratio=config.clip_ratio,
                    value_coef=config.value_coef,
                    entropy_coef=config.entropy_coef,
                    max_grad_norm=config.max_grad_norm,
                    target_kl=config.target_kl,
                    reference_kl_coef=config.reference_kl_coef,
                )
                history["updates"].append({"step": step, **metrics})
                rollout.clear()
            if step % config.eval_interval == 0 or step == config.total_steps:
                summary = evaluate_ppo(
                    agent,
                    list(range(50_000, 50_000 + config.eval_episodes)),
                    size=config.size,
                )
                history["evaluations"].append({"step": step, **summary})
                (config.output_dir / "history.json").write_text(
                    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                label = "GRU" if config.recurrent else "MLP"
                print(f"PPO {label} step={step} success={summary['success_rate']:.0%}")
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
        "algorithm": "doorkey_gru_ppo" if config.recurrent else "doorkey_ppo",
        "seed": config.seed,
        "total_steps": config.total_steps,
        "best_success_rate": best_success,
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DoorKey PPO")
    parser.add_argument("--recurrent", action="store_true")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--count-bonus", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--reference-kl-coef", type=float, default=0.0)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    args = parser.parse_args()
    label = "gru" if args.recurrent else "feedforward"
    if args.curriculum:
        label += "-curriculum"
    if args.count_bonus:
        label += f"-bonus-{args.count_bonus:g}"
    output = args.output_dir or Path(f"runs/doorkey-ppo/{label}-seed-{args.seed}")
    result = train(
        PPOConfig(
            recurrent=args.recurrent,
            size=args.size,
            curriculum=args.curriculum,
            count_bonus=args.count_bonus,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            reference_kl_coef=args.reference_kl_coef,
            initial_checkpoint=args.initial_checkpoint,
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

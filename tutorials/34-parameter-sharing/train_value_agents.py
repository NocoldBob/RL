"""Train IQL, shared DQN, VDN, or QMIX on MPE Simple Spread."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from spread_env import SpreadConfig, assignment, make_env, run_episode, summarize, world_positions
from value_agents import (
    JointReplayBuffer,
    MultiAgentValueLearner,
    ValueConfig,
    padded_state,
    seed_everything,
)


def credit_rewards(
    mode: str,
    environment_rewards: np.ndarray,
    before_agents: np.ndarray,
    before_landmarks: np.ndarray,
    after_agents: np.ndarray,
    after_landmarks: np.ndarray,
) -> np.ndarray:
    if mode == "environment":
        return environment_rewards
    if mode == "shared":
        return np.full_like(environment_rewards, environment_rewards.mean())
    if mode != "assignment_progress":
        raise ValueError(f"unknown reward mode: {mode}")
    targets = assignment(before_agents, before_landmarks)
    progress = np.asarray(
        [
            np.linalg.norm(before_agents[index] - before_landmarks[target])
            - np.linalg.norm(after_agents[index] - after_landmarks[target])
            for index, target in enumerate(targets)
        ],
        dtype=np.float32,
    )
    return np.full_like(progress, environment_rewards.mean()) + 2.0 * progress


def evaluate(learner: MultiAgentValueLearner, config: ValueConfig) -> dict[str, float]:
    environment = make_env(SpreadConfig(agents=config.agents))
    try:
        rows = [
            run_episode(environment, learner.policy, seed=50_000 + index)
            for index in range(config.eval_episodes)
        ]
    finally:
        environment.close()
    return summarize(rows)


def train(config: ValueConfig, reward_mode: str = "environment") -> dict[str, Any]:
    torch.set_num_threads(1)
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    environment = make_env(SpreadConfig(agents=config.agents))
    learner = MultiAgentValueLearner(config)
    replay = JointReplayBuffer(
        config.replay_capacity,
        config.agents,
        learner.observation_size,
    )
    raw_observations, _ = environment.reset(seed=config.seed)
    observations = learner.prepare(raw_observations)
    state = padded_state(environment.state())
    history: dict[str, list[dict[str, Any]]] = {"episodes": [], "updates": [], "evaluations": []}
    episode_return = 0.0
    episode_steps = 0
    best_score = -float("inf")
    try:
        for step in range(1, config.total_steps + 1):
            fraction = min(1.0, step / config.epsilon_decay_steps)
            epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)
            actions = learner.act_array(observations, epsilon)
            names = [f"agent_{index}" for index in range(config.agents)]
            before_agents, before_landmarks = world_positions(environment)
            next_raw, reward_dict, terminations, truncations, _ = environment.step(
                {name: int(actions[index]) for index, name in enumerate(names)}
            )
            after_agents, after_landmarks = world_positions(environment)
            rewards = credit_rewards(
                reward_mode,
                np.asarray([reward_dict[name] for name in names], dtype=np.float32),
                before_agents,
                before_landmarks,
                after_agents,
                after_landmarks,
            )
            done = all(terminations.values()) or all(truncations.values())
            next_observations = (
                learner.prepare(next_raw) if next_raw else np.zeros_like(observations)
            )
            next_state = (
                padded_state(environment.state()) if environment.agents else np.zeros_like(state)
            )
            replay.append(
                observations,
                state,
                actions,
                rewards,
                next_observations,
                next_state,
                done,
            )
            observations = next_observations
            state = next_state
            episode_return += float(np.mean(list(reward_dict.values())))
            episode_steps += 1
            if done:
                history["episodes"].append(
                    {"step": step, "team_return": episode_return, "length": episode_steps}
                )
                raw_observations, _ = environment.reset()
                observations = learner.prepare(raw_observations)
                state = padded_state(environment.state())
                episode_return = 0.0
                episode_steps = 0
            if len(replay) >= max(config.warmup_steps, config.batch_size):
                metrics = learner.update(replay.sample(config.batch_size, learner.device))
                if step % 100 == 0:
                    history["updates"].append({"step": step, **metrics})
            if step % config.target_interval == 0:
                learner.sync_targets()
            if step % config.eval_interval == 0 or step == config.total_steps:
                metrics = evaluate(learner, config)
                history["evaluations"].append({"step": step, "epsilon": epsilon, **metrics})
                (config.output_dir / "history.json").write_text(
                    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                score = metrics["average_team_return"]
                print(
                    f"{config.mode} step={step} return={score:.2f} "
                    f"coverage={metrics['average_coverage']:.0%}"
                )
                if score > best_score:
                    best_score = score
                    learner.save(config.output_dir / "checkpoints" / "best.pt", step)
    finally:
        environment.close()
    learner.save(config.output_dir / "checkpoints" / "latest.pt", config.total_steps)
    summary = {
        "algorithm": config.mode,
        "reward_mode": reward_mode,
        "best_average_team_return": best_score,
        **config.payload(),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("iql", "shared", "vdn", "qmix"), default="iql")
    parser.add_argument("--agent-id", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=("environment", "shared", "assignment_progress"),
        default="environment",
    )
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    label = args.mode + ("-id" if args.agent_id else "") + f"-{args.reward_mode}"
    output = args.output_dir or Path(f"runs/marl-value/{label}-seed-{args.seed}")
    result = train(
        ValueConfig(
            mode=args.mode,
            agent_id=args.agent_id,
            total_steps=args.total_steps,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            output_dir=output,
        ),
        reward_mode=args.reward_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

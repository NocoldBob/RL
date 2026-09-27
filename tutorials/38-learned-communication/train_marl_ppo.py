"""Train IPPO, MAPPO, or a message-sharing MAPPO policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ppo_agents import MultiAgentPPO, MultiAgentRollout, PPOConfig, seed_everything
from spread_env import SpreadConfig, make_env, run_episode, summarize
from value_agents import padded_state


def evaluate(agent: MultiAgentPPO, config: PPOConfig) -> dict[str, float]:
    environment = make_env(SpreadConfig(agents=config.agents))
    try:
        rows = [
            run_episode(environment, agent.policy, seed=50_000 + index)
            for index in range(config.eval_episodes)
        ]
    finally:
        environment.close()
    return summarize(rows)


def train(config: PPOConfig) -> dict[str, Any]:
    torch.set_num_threads(1)
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    environment = make_env(SpreadConfig(agents=config.agents))
    agent = MultiAgentPPO(config)
    rollout = MultiAgentRollout()
    raw_observations, _ = environment.reset(seed=config.seed)
    observations = agent.prepare(raw_observations)
    state = padded_state(environment.state())
    history: dict[str, list[dict[str, Any]]] = {"episodes": [], "updates": [], "evaluations": []}
    episode_return = 0.0
    episode_steps = 0
    best_score = -float("inf")
    try:
        for step in range(1, config.total_steps + 1):
            actions, log_probabilities, values = agent.act(observations, state)
            names = [f"agent_{index}" for index in range(config.agents)]
            next_raw, reward_dict, terminations, truncations, _ = environment.step(
                {name: int(actions[index]) for index, name in enumerate(names)}
            )
            done = all(terminations.values()) or all(truncations.values())
            rewards = np.asarray([reward_dict[name] for name in names], dtype=np.float32)
            next_observations = agent.prepare(next_raw) if next_raw else np.zeros_like(observations)
            next_state = (
                padded_state(environment.state()) if environment.agents else np.zeros_like(state)
            )
            next_values = (
                np.zeros(config.agents, dtype=np.float32)
                if done
                else agent.act(next_observations, next_state)[2]
            )
            rollout.observations.append(observations.copy())
            rollout.states.append(state.copy())
            rollout.actions.append(actions.copy())
            rollout.log_probabilities.append(log_probabilities.copy())
            rollout.rewards.append(rewards)
            rollout.values.append(values.copy())
            rollout.next_values.append(next_values.copy())
            rollout.done.append(done)
            observations, state = next_observations, next_state
            episode_return += float(rewards.mean())
            episode_steps += 1
            if done:
                history["episodes"].append(
                    {"step": step, "team_return": episode_return, "length": episode_steps}
                )
                raw_observations, _ = environment.reset()
                observations = agent.prepare(raw_observations)
                state = padded_state(environment.state())
                episode_return = 0.0
                episode_steps = 0
            if len(rollout) >= config.rollout_steps or step == config.total_steps:
                metrics = agent.update(rollout.build(agent.device, config.gamma, config.gae_lambda))
                history["updates"].append({"step": step, **metrics})
                rollout.clear()
            if step % config.eval_interval == 0 or step == config.total_steps:
                metrics = evaluate(agent, config)
                history["evaluations"].append({"step": step, **metrics})
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
                    agent.save(config.output_dir / "checkpoints" / "best.pt", step)
    finally:
        environment.close()
    agent.save(config.output_dir / "checkpoints" / "latest.pt", config.total_steps)
    summary = {"algorithm": config.mode, "best_average_team_return": best_score, **config.payload()}
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ippo", "mappo", "communication"), default="ippo")
    parser.add_argument("--message-dropout", type=float, default=0.0)
    parser.add_argument("--total-steps", type=int, default=30_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    label = args.mode + (f"-dropout-{args.message_dropout:g}" if args.message_dropout else "")
    output = args.output_dir or Path(f"runs/marl-ppo/{label}-seed-{args.seed}")
    result = train(
        PPOConfig(
            mode=args.mode,
            message_dropout=args.message_dropout,
            total_steps=args.total_steps,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            output_dir=output,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

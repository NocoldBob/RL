"""Train SAC on LunarLanderContinuous-v3."""

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
from lunar_agents import LunarSACAgent, normalize_lunar_observation
from lunar_baselines import ENV_ID
from lunar_evaluation import evaluate_lunar_agent
from lunar_sac_components import ReplayBuffer


@dataclass(slots=True)
class LunarSACConfig:
    total_steps: int = 500_000
    replay_capacity: int = 500_000
    learning_starts: int = 10_000
    batch_size: int = 256
    update_frequency: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    initial_alpha: float = 0.2
    eval_interval: int = 50_000
    eval_episodes: int = 20
    seed: int = 42
    torch_threads: int = 1
    output_dir: Path = Path("runs/lunar-sac/seed-42")

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result


def train_lunar_sac(config: LunarSACConfig) -> dict[str, Any]:
    torch.set_num_threads(config.torch_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = config.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    env = gym.make(ENV_ID)
    agent = LunarSACAgent(
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        initial_alpha=config.initial_alpha,
    )
    replay = ReplayBuffer(config.replay_capacity, 8, 2, seed=config.seed)
    rng = np.random.default_rng(config.seed)
    history: dict[str, list[Any]] = {"episodes": [], "updates": [], "evaluations": []}
    observation, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_steps = 0
    best_return = -float("inf")
    try:
        for step in range(1, config.total_steps + 1):
            action = (
                rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                if step <= config.learning_starts
                else agent.predict(observation, deterministic=False)
            )
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.append(
                normalize_lunar_observation(observation),
                action,
                float(reward) * config.reward_scale,
                normalize_lunar_observation(next_observation),
                terminated,
            )
            episode_return += float(reward)
            episode_steps += 1
            observation = next_observation
            if done:
                history["episodes"].append(
                    {"step": step, "return": episode_return, "length": episode_steps}
                )
                observation, _ = env.reset()
                episode_return = 0.0
                episode_steps = 0
            if (
                step > config.learning_starts
                and step % config.update_frequency == 0
                and len(replay) >= config.batch_size
            ):
                metrics = agent.update(replay.sample(config.batch_size, agent.device))
                if step % 200 == 0:
                    history["updates"].append({"step": step, **metrics})
            if step % config.eval_interval == 0 or step == config.total_steps:
                summary, _, _ = evaluate_lunar_agent(
                    agent, list(range(50_000, 50_000 + config.eval_episodes))
                )
                history["evaluations"].append({"step": step, **summary})
                (config.output_dir / "history.json").write_text(
                    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    f"SAC seed={config.seed} step={step} "
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
        "algorithm": "lunar_sac",
        "seed": config.seed,
        "total_steps": config.total_steps,
        "best_eval_return": best_return,
        "final_alpha": float(agent.alpha.detach()),
    }
    (config.output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on LunarLanderContinuous-v3")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(f"runs/lunar-sac/seed-{args.seed}")
    config = LunarSACConfig(
        total_steps=args.total_steps,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        output_dir=output,
    )
    print(json.dumps(train_lunar_sac(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Train and fairly compare five Snake policies on shared evaluation seeds."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dqn import DQNAgent
from environment import SnakeEnv
from main import TrainConfig, as_tensor, choose_device, train
from model import ConvActorCritic
from ppo import PPOAgent
from train_dqn import DQNTrainConfig, train_dqn
from train_ppo import PPOTrainConfig, train_ppo

Policy = Callable[[np.ndarray, SnakeEnv], int]


def evaluate_policy(
    policy: Policy,
    *,
    grid_size: int,
    end_score: int,
    max_steps: int,
    episodes: int,
    seed_base: int,
) -> dict[str, float]:
    env = SnakeEnv(grid_size=grid_size, end_score=end_score, max_steps=max_steps)
    rewards: list[float] = []
    scores: list[int] = []
    successes = 0
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed_base + episode)
            episode_reward = 0.0
            while True:
                action = policy(observation, env)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    rewards.append(episode_reward)
                    scores.append(int(info["score"]))
                    successes += int(info["length"] >= end_score)
                    break
    finally:
        env.close()
    return {
        "reward": float(np.mean(rewards)),
        "score": float(np.mean(scores)),
        "success_rate": successes / episodes,
    }


def load_actor_critic(checkpoint_path: Path, device: torch.device) -> ConvActorCritic:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = dict(checkpoint.get("config", {}))
    grid_size = int(config.get("grid_size", 6))
    env = SnakeEnv(grid_size=grid_size)
    model = ConvActorCritic(
        input_channels=env.observation_space.shape[0],
        output_dim=env.action_space.n,
        grid_size=grid_size,
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
        entropy_coef=float(config.get("entropy_coef", 0.01)),
    ).to(device)
    env.close()
    model.load_checkpoint(checkpoint_path)
    model.eval()
    return model


def load_dqn(checkpoint_path: Path, device: torch.device) -> DQNAgent:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = dict(checkpoint.get("config", {}))
    grid_size = int(config.get("grid_size", 6))
    env = SnakeEnv(grid_size=grid_size)
    agent = DQNAgent(
        input_channels=env.observation_space.shape[0],
        action_count=env.action_space.n,
        grid_size=grid_size,
        learning_rate=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        device=device,
    )
    env.close()
    agent.load_checkpoint(checkpoint_path)
    agent.online.eval()
    return agent


def load_ppo(checkpoint_path: Path, device: torch.device) -> PPOAgent:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = dict(checkpoint.get("config", {}))
    grid_size = int(config.get("grid_size", 6))
    env = SnakeEnv(grid_size=grid_size)
    agent = PPOAgent(
        input_channels=env.observation_space.shape[0],
        action_count=env.action_space.n,
        grid_size=grid_size,
        learning_rate=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        device=device,
    )
    env.close()
    agent.load_checkpoint(checkpoint_path)
    agent.model.eval()
    return agent


def timed_training(function: Callable[[], Any], timing_path: Path, *, reuse: bool) -> float:
    if reuse and timing_path.is_file():
        return float(json.loads(timing_path.read_text(encoding="utf-8"))["seconds"])
    started = time.perf_counter()
    function()
    seconds = time.perf_counter() - started
    timing_path.write_text(
        json.dumps({"seconds": seconds}, indent=2),
        encoding="utf-8",
    )
    return seconds


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for strategy in ("random", "teacher", "actor_critic", "dqn", "ppo"):
        selected = [row for row in rows if row["strategy"] == strategy]
        values: dict[str, float] = {}
        for metric in ("reward", "score", "success_rate", "train_seconds"):
            samples = [float(row[metric]) for row in selected]
            values[f"{metric}_mean"] = statistics.mean(samples)
            values[f"{metric}_std"] = statistics.pstdev(samples)
        aggregate[strategy] = values
    return aggregate


def run_benchmark(
    *,
    seeds: list[int],
    episodes: int,
    eval_episodes: int,
    grid_size: int,
    end_score: int,
    max_steps: int,
    output_dir: Path,
    device_name: str,
    torch_threads: int,
    reuse: bool,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if episodes < 1 or eval_episodes < 1:
        raise ValueError("episode counts must be positive")
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(torch_threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(device_name)
    rows: list[dict[str, Any]] = []
    evaluation_seed_base = 300_000

    for seed in seeds:
        seed_dir = output_dir / f"seed-{seed}"
        ac_dir = seed_dir / "actor-critic"
        dqn_dir = seed_dir / "dqn"
        ppo_dir = seed_dir / "ppo"
        ac_checkpoint = ac_dir / "checkpoints" / "latest.pt"
        dqn_checkpoint = dqn_dir / "checkpoints" / "latest.pt"
        ppo_checkpoint = ppo_dir / "checkpoints" / "latest.pt"

        ac_seconds = timed_training(
            lambda current_seed=seed, current_dir=ac_dir: train(
                TrainConfig(
                    episodes=episodes,
                    teacher_episodes=0,
                    grid_size=grid_size,
                    end_score=end_score,
                    max_steps=max_steps,
                    eval_interval=0,
                    save_interval=0,
                    seed=current_seed,
                    device=device_name,
                    output_dir=current_dir,
                    tensorboard=False,
                )
            ),
            ac_dir / "timing.json",
            reuse=reuse and ac_checkpoint.is_file(),
        )
        dqn_seconds = timed_training(
            lambda current_seed=seed, current_dir=dqn_dir: train_dqn(
                DQNTrainConfig(
                    episodes=episodes,
                    grid_size=grid_size,
                    end_score=end_score,
                    max_steps=max_steps,
                    eval_interval=0,
                    save_interval=0,
                    seed=current_seed,
                    torch_threads=torch_threads,
                    device=device_name,
                    output_dir=current_dir,
                    tensorboard=False,
                )
            ),
            dqn_dir / "timing.json",
            reuse=reuse and dqn_checkpoint.is_file(),
        )
        ppo_seconds = timed_training(
            lambda current_seed=seed, current_dir=ppo_dir: train_ppo(
                PPOTrainConfig(
                    episodes=episodes,
                    grid_size=grid_size,
                    end_score=end_score,
                    max_steps=max_steps,
                    eval_interval=0,
                    save_interval=0,
                    seed=current_seed,
                    torch_threads=torch_threads,
                    device=device_name,
                    output_dir=current_dir,
                    tensorboard=False,
                )
            ),
            ppo_dir / "timing.json",
            reuse=reuse and ppo_checkpoint.is_file(),
        )

        actor_critic = load_actor_critic(ac_checkpoint, device)
        dqn = load_dqn(dqn_checkpoint, device)
        ppo = load_ppo(ppo_checkpoint, device)
        random_source = random.Random(seed)
        ppo_generator = torch.Generator(device=device)
        ppo_generator.manual_seed(seed)
        policies: dict[str, tuple[Policy, float]] = {
            "random": (
                lambda _observation, env, rng=random_source: rng.randrange(env.action_space.n),
                0.0,
            ),
            "teacher": (lambda _observation, env: env.teacher_action(), 0.0),
            "actor_critic": (
                lambda observation, _env, model=actor_critic: model.predict(
                    as_tensor(observation, device),
                    deterministic=True,
                ),
                ac_seconds,
            ),
            "dqn": (
                lambda observation, _env, agent=dqn, rng=random_source: agent.select_action(
                    observation,
                    epsilon=0.0,
                    random_source=rng,
                ),
                dqn_seconds,
            ),
            "ppo": (
                lambda observation, _env, agent=ppo, generator=ppo_generator: agent.predict(
                    observation,
                    generator=generator,
                ),
                ppo_seconds,
            ),
        }
        for strategy, (policy, train_seconds) in policies.items():
            metrics = evaluate_policy(
                policy,
                grid_size=grid_size,
                end_score=end_score,
                max_steps=max_steps,
                episodes=eval_episodes,
                seed_base=evaluation_seed_base,
            )
            row = {
                "seed": seed,
                "strategy": strategy,
                "train_seconds": train_seconds,
                **metrics,
            }
            rows.append(row)
            print(
                f"seed={seed} strategy={strategy} reward={metrics['reward']:.2f} "
                f"score={metrics['score']:.2f} success={metrics['success_rate']:.0%}"
            )

    result = {
        "config": {
            "seeds": seeds,
            "train_episodes": episodes,
            "eval_episodes": eval_episodes,
            "grid_size": grid_size,
            "end_score": end_score,
            "max_steps": max_steps,
            "evaluation_seed_base": evaluation_seed_base,
            "device": str(device),
            "torch_threads": torch_threads,
        },
        "runs": rows,
        "aggregate": aggregate_results(rows),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Snake policies fairly")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 2026])
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--end-score", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--reuse", action="store_true")
    arguments = parser.parse_args()
    result = run_benchmark(
        seeds=arguments.seeds,
        episodes=arguments.episodes,
        eval_episodes=arguments.eval_episodes,
        grid_size=arguments.grid_size,
        end_score=arguments.end_score,
        max_steps=arguments.max_steps,
        output_dir=arguments.output_dir,
        device_name=arguments.device,
        torch_threads=arguments.torch_threads,
        reuse=arguments.reuse,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

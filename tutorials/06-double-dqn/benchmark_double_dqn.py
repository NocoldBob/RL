"""Paired DQN/Double DQN benchmark with Q-value diagnostics."""

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
from benchmark import load_dqn
from dqn import DQNAgent
from environment import SnakeEnv
from main import choose_device
from train_dqn import DQNTrainConfig, train_dqn


def timed_training(function: Callable[[], Any], timing_path: Path, *, reuse: bool) -> float:
    if reuse and timing_path.is_file():
        return float(json.loads(timing_path.read_text(encoding="utf-8"))["seconds"])
    started = time.perf_counter()
    function()
    seconds = time.perf_counter() - started
    timing_path.write_text(json.dumps({"seconds": seconds}, indent=2), encoding="utf-8")
    return seconds


@torch.no_grad()
def evaluate_q_diagnostics(
    agent: DQNAgent,
    *,
    grid_size: int,
    end_score: int,
    max_steps: int,
    episodes: int,
    seed_base: int,
    gamma: float,
) -> dict[str, float]:
    """Compare initial Q predictions with realized greedy-policy returns."""

    env = SnakeEnv(grid_size=grid_size, end_score=end_score, max_steps=max_steps)
    random_source = random.Random(seed_base)
    rewards: list[float] = []
    scores: list[int] = []
    initial_q_values: list[float] = []
    discounted_returns: list[float] = []
    visited_states: list[np.ndarray] = []
    successes = 0
    online_was_training = agent.online.training
    target_was_training = agent.target.training
    agent.online.eval()
    agent.target.eval()
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed_base + episode)
            initial_state = (
                torch.from_numpy(observation)
                .unsqueeze(0)
                .to(
                    device=agent.device,
                    dtype=torch.float32,
                )
            )
            initial_q_values.append(float(agent.online(initial_state).max().item()))
            episode_reward = 0.0
            discounted_return = 0.0
            discount = 1.0
            while True:
                visited_states.append(observation.copy())
                action = agent.select_action(
                    observation,
                    epsilon=0.0,
                    random_source=random_source,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                discounted_return += discount * reward
                discount *= gamma
                if terminated or truncated:
                    rewards.append(episode_reward)
                    discounted_returns.append(discounted_return)
                    scores.append(int(info["score"]))
                    successes += int(info["length"] >= end_score)
                    break
    finally:
        env.close()
        agent.online.train(online_was_training)
        agent.target.train(target_was_training)

    selection_gaps: list[float] = []
    disagreements = 0
    for start in range(0, len(visited_states), 512):
        states = torch.from_numpy(np.stack(visited_states[start : start + 512])).to(
            device=agent.device,
            dtype=torch.float32,
        )
        online_actions = agent.online(states).argmax(dim=1, keepdim=True)
        target_q_values = agent.target(states)
        target_actions = target_q_values.argmax(dim=1, keepdim=True)
        standard_values = target_q_values.max(dim=1).values
        double_values = target_q_values.gather(1, online_actions).squeeze(1)
        selection_gaps.extend((standard_values - double_values).cpu().tolist())
        disagreements += int((online_actions != target_actions).sum().item())

    q_array = np.asarray(initial_q_values, dtype=np.float64)
    return_array = np.asarray(discounted_returns, dtype=np.float64)
    q_return_gaps = q_array - return_array
    return {
        "reward": float(np.mean(rewards)),
        "score": float(np.mean(scores)),
        "success_rate": successes / episodes,
        "initial_q_mean": float(q_array.mean()),
        "discounted_return_mean": float(return_array.mean()),
        "q_return_gap_mean": float(q_return_gaps.mean()),
        "q_return_gap_abs_mean": float(np.abs(q_return_gaps).mean()),
        "q_overestimate_rate": float(np.mean(q_return_gaps > 0.0)),
        "target_selection_gap_mean": float(np.mean(selection_gaps)),
        "action_disagreement_rate": disagreements / len(visited_states),
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = (
        "reward",
        "score",
        "success_rate",
        "train_seconds",
        "initial_q_mean",
        "discounted_return_mean",
        "q_return_gap_mean",
        "q_return_gap_abs_mean",
        "q_overestimate_rate",
        "target_selection_gap_mean",
        "action_disagreement_rate",
    )
    aggregate: dict[str, dict[str, float]] = {}
    for algorithm in ("dqn", "double_dqn"):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        values: dict[str, float] = {}
        for metric in metrics:
            samples = [float(row[metric]) for row in selected]
            values[f"{metric}_mean"] = statistics.mean(samples)
            values[f"{metric}_std"] = statistics.pstdev(samples)
        aggregate[algorithm] = values
    return aggregate


def run_benchmark(
    *,
    seeds: list[int],
    episodes: int,
    eval_episodes: int,
    grid_size: int,
    end_score: int,
    max_steps: int,
    gamma: float,
    output_dir: Path,
    device_name: str,
    torch_threads: int,
    reuse: bool,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if episodes < 1 or eval_episodes < 1 or torch_threads < 1:
        raise ValueError("episode counts and torch_threads must be positive")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be between zero and one")

    torch.set_num_threads(torch_threads)
    device = choose_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    evaluation_seed_base = 400_000

    for seed in seeds:
        for algorithm, double_dqn in (("dqn", False), ("double_dqn", True)):
            run_dir = output_dir / f"seed-{seed}" / algorithm.replace("_", "-")
            checkpoint = run_dir / "checkpoints" / "latest.pt"
            seconds = timed_training(
                lambda current_seed=seed, current_dir=run_dir, use_double=double_dqn: train_dqn(
                    DQNTrainConfig(
                        episodes=episodes,
                        grid_size=grid_size,
                        end_score=end_score,
                        max_steps=max_steps,
                        gamma=gamma,
                        double_dqn=use_double,
                        eval_interval=0,
                        save_interval=0,
                        seed=current_seed,
                        torch_threads=torch_threads,
                        device=device_name,
                        output_dir=current_dir,
                        tensorboard=False,
                    )
                ),
                run_dir / "timing.json",
                reuse=reuse and checkpoint.is_file(),
            )
            agent = load_dqn(checkpoint, device)
            diagnostics = evaluate_q_diagnostics(
                agent,
                grid_size=grid_size,
                end_score=end_score,
                max_steps=max_steps,
                episodes=eval_episodes,
                seed_base=evaluation_seed_base,
                gamma=gamma,
            )
            row = {
                "seed": seed,
                "algorithm": algorithm,
                "train_seconds": seconds,
                **diagnostics,
            }
            rows.append(row)
            print(
                f"seed={seed} algorithm={algorithm} reward={diagnostics['reward']:.2f} "
                f"score={diagnostics['score']:.2f} success={diagnostics['success_rate']:.0%} "
                f"q-gap={diagnostics['q_return_gap_mean']:.2f}"
            )

    result = {
        "config": {
            "seeds": seeds,
            "train_episodes": episodes,
            "eval_episodes": eval_episodes,
            "grid_size": grid_size,
            "end_score": end_score,
            "max_steps": max_steps,
            "gamma": gamma,
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
    parser = argparse.ArgumentParser(description="Compare DQN and Double DQN fairly")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 2026])
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--end-score", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/double-dqn-benchmark"),
    )
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
        gamma=arguments.gamma,
        output_dir=arguments.output_dir,
        device_name=arguments.device,
        torch_threads=arguments.torch_threads,
        reuse=arguments.reuse,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Four-way DQN benchmark for isolating Dueling and Double DQN effects."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark import load_dqn
from benchmark_double_dqn import evaluate_q_diagnostics, timed_training
from dqn import DQNAgent, DuelingQNetwork
from environment import SnakeEnv
from main import choose_device
from train_dqn import DQNTrainConfig, train_dqn

VARIANTS = (
    ("dqn", False, False),
    ("double_dqn", True, False),
    ("dueling_dqn", False, True),
    ("dueling_double_dqn", True, True),
)


@torch.no_grad()
def evaluate_representation_diagnostics(
    agent: DQNAgent,
    *,
    grid_size: int,
    end_score: int,
    max_steps: int,
    episodes: int,
    seed_base: int,
) -> dict[str, float | int | None]:
    """Measure action separation and, when available, Dueling head outputs."""

    env = SnakeEnv(grid_size=grid_size, end_score=end_score, max_steps=max_steps)
    random_source = random.Random(seed_base)
    visited_states: list[np.ndarray] = []
    was_training = agent.online.training
    agent.online.eval()
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed_base + episode)
            while True:
                visited_states.append(observation.copy())
                action = agent.select_action(
                    observation,
                    epsilon=0.0,
                    random_source=random_source,
                )
                observation, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()
        agent.online.train(was_training)

    q_ranges: list[float] = []
    q_margins: list[float] = []
    state_values: list[float] = []
    advantage_magnitudes: list[float] = []
    advantage_ranges: list[float] = []
    for start in range(0, len(visited_states), 512):
        states = torch.from_numpy(np.stack(visited_states[start : start + 512])).to(
            device=agent.device,
            dtype=torch.float32,
        )
        q_values = agent.online(states)
        q_ranges.extend((q_values.max(dim=1).values - q_values.min(dim=1).values).cpu().tolist())
        top_two = q_values.topk(k=2, dim=1).values
        q_margins.extend((top_two[:, 0] - top_two[:, 1]).cpu().tolist())
        if isinstance(agent.online, DuelingQNetwork):
            value, advantage = agent.online.decompose(states)
            state_values.extend(value.squeeze(1).cpu().tolist())
            advantage_magnitudes.extend(advantage.abs().mean(dim=1).cpu().tolist())
            advantage_ranges.extend(
                (advantage.max(dim=1).values - advantage.min(dim=1).values).cpu().tolist()
            )

    result: dict[str, float | int | None] = {
        "parameter_count": sum(parameter.numel() for parameter in agent.online.parameters()),
        "visited_state_count": len(visited_states),
        "q_action_range_mean": float(np.mean(q_ranges)),
        "q_action_margin_mean": float(np.mean(q_margins)),
        "state_value_mean": None,
        "state_value_std": None,
        "advantage_abs_mean": None,
        "advantage_range_mean": None,
    }
    if state_values:
        result.update(
            {
                "state_value_mean": float(np.mean(state_values)),
                "state_value_std": float(np.std(state_values)),
                "advantage_abs_mean": float(np.mean(advantage_magnitudes)),
                "advantage_range_mean": float(np.mean(advantage_ranges)),
            }
        )
    return result


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
        "parameter_count",
        "visited_state_count",
        "q_action_range_mean",
        "q_action_margin_mean",
        "state_value_mean",
        "state_value_std",
        "advantage_abs_mean",
        "advantage_range_mean",
    )
    aggregate: dict[str, dict[str, float]] = {}
    for algorithm, _double_dqn, _dueling in VARIANTS:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        values: dict[str, float] = {}
        for metric in metrics:
            samples = [float(row[metric]) for row in selected if row[metric] is not None]
            if samples:
                values[f"{metric}_mean"] = statistics.mean(samples)
                values[f"{metric}_std"] = statistics.pstdev(samples)
        aggregate[algorithm] = values
    return aggregate


def factorial_effects(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Report paired main effects and interaction for the headline metrics."""

    metrics = ("reward", "score", "success_rate", "q_action_margin_mean")
    by_seed = {(row["seed"], row["algorithm"]): row for row in rows}
    effects: dict[str, dict[str, float]] = {}
    for metric in metrics:
        dueling_effects: list[float] = []
        double_effects: list[float] = []
        interactions: list[float] = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            dqn = float(by_seed[(seed, "dqn")][metric])
            double = float(by_seed[(seed, "double_dqn")][metric])
            dueling = float(by_seed[(seed, "dueling_dqn")][metric])
            combined = float(by_seed[(seed, "dueling_double_dqn")][metric])
            dueling_effects.append(((dueling - dqn) + (combined - double)) / 2.0)
            double_effects.append(((double - dqn) + (combined - dueling)) / 2.0)
            interactions.append((combined - dueling) - (double - dqn))
        effects[metric] = {
            "dueling_main_effect_mean": statistics.mean(dueling_effects),
            "double_main_effect_mean": statistics.mean(double_effects),
            "interaction_mean": statistics.mean(interactions),
        }
    return effects


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
    evaluation_seed_base = 500_000

    for seed in seeds:
        for algorithm, double_dqn, dueling in VARIANTS:
            run_dir = output_dir / f"seed-{seed}" / algorithm.replace("_", "-")
            checkpoint = run_dir / "checkpoints" / "latest.pt"
            training_config = DQNTrainConfig(
                episodes=episodes,
                grid_size=grid_size,
                end_score=end_score,
                max_steps=max_steps,
                gamma=gamma,
                double_dqn=double_dqn,
                dueling=dueling,
                eval_interval=0,
                save_interval=0,
                seed=seed,
                torch_threads=torch_threads,
                device=device_name,
                output_dir=run_dir,
                tensorboard=False,
            )
            seconds = timed_training(
                lambda current_config=training_config: train_dqn(current_config),
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
            representation = evaluate_representation_diagnostics(
                agent,
                grid_size=grid_size,
                end_score=end_score,
                max_steps=max_steps,
                episodes=eval_episodes,
                seed_base=evaluation_seed_base,
            )
            row = {
                "seed": seed,
                "algorithm": algorithm,
                "double_dqn": double_dqn,
                "dueling": dueling,
                "train_seconds": seconds,
                **diagnostics,
                **representation,
            }
            rows.append(row)
            print(
                f"seed={seed} algorithm={algorithm} reward={diagnostics['reward']:.2f} "
                f"score={diagnostics['score']:.2f} success={diagnostics['success_rate']:.0%} "
                f"margin={representation['q_action_margin_mean']:.3f}"
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
        "factorial_effects": factorial_effects(rows),
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
    parser = argparse.ArgumentParser(description="Compare four DQN variants fairly")
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
        default=Path("runs/dueling-dqn-benchmark"),
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

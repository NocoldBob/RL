"""Train several SAC seeds and compare them with continuous-control baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from mountain_car_baselines import ENV_ID
from sac import SACAgent
from train_sac import SACConfig, evaluate_sac_agent, train_sac
from visualize_sac import generate_assets


def aggregate_training_seeds(seed_results: list[dict[str, Any]]) -> dict[str, float]:
    names = (
        "average_return",
        "success_rate",
        "average_steps",
        "average_success_steps",
        "average_action_cost",
    )
    values = {
        name: [row["summary"][name] for row in seed_results if row["summary"][name] is not None]
        for name in names
    }
    return {f"{name}_mean": float(np.mean(values[name])) for name in names} | {
        f"{name}_std": float(np.std(values[name])) for name in names
    }


def exploration_probe(
    repeats: list[int], *, episodes: int = 100, base_seed: int = 10_000
) -> list[dict[str, float | int]]:
    """Measure whether temporally correlated random actions can discover the sparse reward."""
    env = gym.make(ENV_ID)
    results = []
    try:
        for repeat in repeats:
            successes = 0
            returns = []
            for seed in range(base_seed, base_seed + episodes):
                env.reset(seed=seed)
                rng = np.random.default_rng(seed)
                action = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
                terminated = truncated = False
                episode_return = 0.0
                step = 0
                while not (terminated or truncated):
                    if step % repeat == 0:
                        action = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
                    _, reward, terminated, truncated, _ = env.step(action)
                    episode_return += float(reward)
                    step += 1
                successes += int(terminated)
                returns.append(episode_return)
            results.append(
                {
                    "action_repeat": repeat,
                    "episodes": episodes,
                    "success_rate": successes / episodes,
                    "average_return": float(np.mean(returns)),
                }
            )
    finally:
        env.close()
    return results


def run_benchmark(
    training_seeds: list[int],
    total_steps: int,
    evaluation_seeds: list[int],
    runs_dir: Path,
    baseline_path: Path,
    *,
    trace_seed: int = 42,
) -> dict[str, Any]:
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    seed_results: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}

    for seed in training_seeds:
        run_dir = runs_dir / f"seed-{seed}"
        config = SACConfig(total_steps=total_steps, seed=seed, output_dir=run_dir)
        train_sac(config)
        agent = SACAgent()
        checkpoint = run_dir / "checkpoints" / "best.pt"
        agent.load_checkpoint(checkpoint)
        summary, rows, _ = evaluate_sac_agent(agent, evaluation_seeds)
        _, _, trace = evaluate_sac_agent(agent, [trace_seed], capture_seed=trace_seed)
        seed_results.append(
            {
                "training_seed": seed,
                "checkpoint": str(checkpoint),
                "summary": summary,
                "trace": trace,
            }
        )
        episode_rows.extend({"training_seed": seed, **row} for row in rows)
        histories[str(seed)] = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))

    return {
        "schema_version": 1,
        "environment": baseline_report["environment"],
        "experiment": {
            "algorithm": "sac",
            "training_seeds": training_seeds,
            "total_steps_per_seed": total_steps,
            "evaluation_seeds": evaluation_seeds,
            "deterministic_evaluation": True,
            "baseline_source": str(baseline_path),
            "reward_shaping": False,
            "teacher_demonstrations": False,
        },
        "exploration_probe": exploration_probe([1, 5, 10, 20]),
        "sac": {
            "aggregate_across_training_seeds": aggregate_training_seeds(seed_results),
            "training_seeds": seed_results,
            "episodes": episode_rows,
            "histories": histories,
        },
        "baselines": baseline_report["aggregate"],
    }


def write_report(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = report["sac"]["episodes"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark SAC on continuous Mountain Car")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[7, 42, 2026])
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-base-seed", type=int, default=2026)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/sac"))
    parser.add_argument(
        "--baseline-json", type=Path, default=Path("docs/experiments/09-continuous-baselines.json")
    )
    parser.add_argument("--output-json", type=Path, default=Path("docs/experiments/11-sac.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("docs/experiments/11-sac.csv"))
    parser.add_argument("--assets-dir", type=Path, default=Path("docs/assets/csdn-11"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluation_seeds = list(
        range(args.evaluation_base_seed, args.evaluation_base_seed + args.evaluation_episodes)
    )
    report = run_benchmark(
        args.training_seeds,
        args.total_steps,
        evaluation_seeds,
        args.runs_dir,
        args.baseline_json,
    )
    write_report(report, args.output_json, args.output_csv)
    generate_assets(report, args.assets_dir)
    print(json.dumps(report["sac"]["aggregate_across_training_seeds"], indent=2))


if __name__ == "__main__":
    main()

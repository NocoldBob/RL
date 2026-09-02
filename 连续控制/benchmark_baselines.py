"""Benchmark simple policies on MountainCarContinuous-v0."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
from mountain_car_baselines import (
    ENV_ID,
    POLICY_LABELS,
    POLICY_NAMES,
    evaluate_baselines,
    run_episode,
)
from visualize_baselines import generate_assets


def build_report(episodes: int, base_seed: int, trace_seed: int) -> dict[str, Any]:
    seeds = list(range(base_seed, base_seed + episodes))
    aggregate, rows = evaluate_baselines(seeds)
    traces = {
        policy: run_episode(policy, trace_seed, capture_trace=True).trace for policy in POLICY_NAMES
    }
    env = gym.make(ENV_ID)
    try:
        environment = {
            "id": ENV_ID,
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "action_meaning": "A scalar throttle in [-1, 1].",
            "reward": "Goal bonus 100 minus 0.1 * action^2 per step.",
            "time_limit": int(env.spec.max_episode_steps or 0),
        }
    finally:
        env.close()
    return {
        "schema_version": 1,
        "environment": environment,
        "experiment": {
            "episodes_per_policy": episodes,
            "base_seed": base_seed,
            "evaluation_seeds": seeds,
            "trace_seed": trace_seed,
            "policies": [{"name": name, "label": POLICY_LABELS[name]} for name in POLICY_NAMES],
        },
        "aggregate": aggregate,
        "episodes": [row.metrics() for row in rows],
        "traces": traces,
    }


def write_report(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    episode_rows = report["episodes"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare four continuous-control baselines")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--trace-seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/experiments/09-continuous-baselines.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("docs/experiments/09-continuous-baselines.csv"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("docs/assets/csdn-09"),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.episodes <= 0:
        raise ValueError("episodes must be positive")
    report = build_report(arguments.episodes, arguments.base_seed, arguments.trace_seed)
    write_report(report, arguments.output_json, arguments.output_csv)
    generate_assets(report, arguments.assets_dir)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

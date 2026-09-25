"""Benchmark LunarLander continuous-action baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from lunar_baselines import ENV_ID, POLICY_NAMES, run_episode
from visualize_lunar_baselines import generate_assets


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "episodes": len(rows),
        "average_return": float(np.mean([row["episode_return"] for row in rows])),
        "return_std": float(np.std([row["episode_return"] for row in rows])),
        "landing_rate": float(np.mean([row["landed"] for row in rows])),
        "crash_rate": float(np.mean([row["crashed"] for row in rows])),
        "timeout_rate": float(np.mean([row["timed_out"] for row in rows])),
        "average_steps": float(np.mean([row["steps"] for row in rows])),
        "average_main_engine_usage": float(np.mean([row["main_engine_usage"] for row in rows])),
        "average_side_engine_usage": float(np.mean([row["side_engine_usage"] for row in rows])),
        "average_final_speed": float(np.mean([row["final_speed"] for row in rows])),
    }


def build_report(
    episodes: int = 100,
    base_seed: int = 2026,
    *,
    trace_seed: int = 42,
) -> dict[str, Any]:
    env = gym.make(ENV_ID)
    try:
        environment = {
            "id": ENV_ID,
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "action_meaning": "[main engine throttle, lateral engine throttle]",
            "solved_score": 200,
        }
    finally:
        env.close()

    rows = []
    traces = {}
    for policy in POLICY_NAMES:
        for seed in range(base_seed, base_seed + episodes):
            result, _ = run_episode(policy, seed)
            rows.append(result.payload())
        _, trace = run_episode(policy, trace_seed, capture_trace=True)
        traces[policy] = trace

    grouped = {
        policy: aggregate([row for row in rows if row["policy"] == policy])
        for policy in POLICY_NAMES
    }
    return {
        "schema_version": 1,
        "environment": environment,
        "experiment": {
            "episodes_per_policy": episodes,
            "base_seed": base_seed,
            "evaluation_seeds": list(range(base_seed, base_seed + episodes)),
            "trace_seed": trace_seed,
        },
        "aggregate": grouped,
        "episodes": rows,
        "traces": traces,
    }


def write_report(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = report["episodes"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LunarLander continuous baselines")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--trace-seed", type=int, default=42)
    parser.add_argument(
        "--output-json", type=Path, default=Path("docs/experiments/13-lunar-baselines.json")
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("docs/experiments/13-lunar-baselines.csv")
    )
    parser.add_argument("--assets-dir", type=Path, default=Path("docs/assets/csdn-13"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(args.episodes, args.base_seed, trace_seed=args.trace_seed)
    write_report(report, args.output_json, args.output_csv)
    generate_assets(report, args.assets_dir)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Benchmark random, forward-only, and expert DoorKey policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from doorkey_env import POLICIES, run_episode


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "key_pickup_rate": float(np.mean([row["picked_key"] for row in rows])),
        "door_open_rate": float(np.mean([row["opened_door"] for row in rows])),
        "average_return": float(np.mean([row["episode_return"] for row in rows])),
        "average_steps": float(np.mean([row["steps"] for row in rows])),
    }


def benchmark(sizes: list[int], seeds: list[int]) -> dict[str, Any]:
    rows = []
    traces = {}
    for size in sizes:
        for policy in POLICIES:
            for seed in seeds:
                result, trace = run_episode(
                    policy, size, seed, capture_trace=policy == "expert" and seed == seeds[0]
                )
                rows.append(result.payload())
                if trace is not None:
                    traces[f"{size}x{size}-{policy}"] = trace
    groups = {}
    for size in sizes:
        for policy in POLICIES:
            selected = [row for row in rows if row["size"] == size and row["policy"] == policy]
            groups[f"{size}x{size}-{policy}"] = summarize(selected)
    return {
        "experiment": "lesson-18-doorkey-baselines",
        "sizes": sizes,
        "seeds": seeds,
        "groups": groups,
        "episodes": rows,
        "expert_traces": traces,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DoorKey baselines")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 8])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/18-doorkey-baselines.json")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("docs/experiments/18-doorkey-baselines.csv")
    )
    args = parser.parse_args()
    report = benchmark(args.sizes, list(range(args.episodes)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report["episodes"], args.csv_output)
    print(json.dumps(report["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

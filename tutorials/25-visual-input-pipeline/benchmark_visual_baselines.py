"""Benchmark simple CarRacing policies and the image preprocessing pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from carracing_env import (
    DrivingConfig,
    VisualCarRacing,
    privileged_track_action,
    run_episode,
    summarize_episodes,
)


def benchmark(episodes: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    configurations = {
        "idle": lambda _obs: np.asarray([0.0, -1.0, -1.0], dtype=np.float32),
        "straight": lambda _obs: np.asarray([0.0, 0.4, -1.0], dtype=np.float32),
        "random": lambda _obs: rng.uniform(-1.0, 1.0, 3).astype(np.float32),
    }
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float]] = {}
    for name, policy in configurations.items():
        environment = VisualCarRacing(DrivingConfig(frame_stack=4, seed=seed))
        try:
            rows = [
                run_episode(environment, policy, seed=seed + index) for index in range(episodes)
            ]
        finally:
            environment.close()
        summaries[name] = summarize_episodes(rows)
        all_rows.extend({"policy": name, **row} for row in rows)
    environment = VisualCarRacing(DrivingConfig(frame_stack=4, seed=seed))
    try:
        rows = [
            run_episode(
                environment,
                lambda _observation: privileged_track_action(environment),
                seed=seed + index,
            )
            for index in range(episodes)
        ]
    finally:
        environment.close()
    summaries["privileged_local_follower"] = summarize_episodes(rows)
    all_rows.extend({"policy": "privileged_local_follower", **row} for row in rows)
    return {
        "experiment": "25-visual-input-pipeline",
        "episodes_per_policy": episodes,
        "seed": seed,
        "preprocessing": {
            "source_shape": [96, 96, 3],
            "crop": "top 84 rows",
            "target_shape": [4, 64, 64],
            "grayscale": True,
            "frame_skip": 4,
        },
        "summaries": summaries,
        "episodes": all_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/25-visual-baselines.json")
    )
    args = parser.parse_args()
    report = benchmark(args.episodes, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["episodes"][0]))
        writer.writeheader()
        writer.writerows(report["episodes"])
    print(json.dumps(report["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

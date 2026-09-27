"""Evaluate random, greedy-nearest, and coordinated-assignment baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from spread_env import (
    SpreadConfig,
    assignment_policy,
    make_env,
    nearest_policy,
    random_policy,
    run_episode,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/32-spread-baselines.json")
    )
    args = parser.parse_args()
    policies = {"random": random_policy, "nearest": nearest_policy, "assignment": assignment_policy}
    all_rows = []
    summaries = {}
    for name, policy in policies.items():
        environment = make_env(SpreadConfig(agents=args.agents))
        try:
            rows = [
                run_episode(environment, policy, seed=10_000 + index)
                for index in range(args.episodes)
            ]
        finally:
            environment.close()
        summaries[name] = summarize(rows)
        all_rows.extend({"policy": name, **row} for row in rows)
    report = {
        "experiment": "32-spread-baselines",
        "agents": args.agents,
        "episodes_per_policy": args.episodes,
        "summaries": summaries,
        "episodes": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

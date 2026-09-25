"""Compare sparse and milestone-shaped full-map DoorKey DQN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from doorkey_dqn import DoorKeyDQNAgent, evaluate_dqn


def build_report(runs_root: Path, episodes: int) -> dict[str, Any]:
    variants = []
    histories = {}
    seeds = list(range(60_000, 60_000 + episodes))
    for reward_mode in ("sparse", "milestones"):
        run_dir = runs_root / f"{reward_mode}-seed-42"
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        agent = DoorKeyDQNAgent(5, 5)
        checkpoint = agent.load(run_dir / "checkpoints" / "best.pt")
        summary = evaluate_dqn(agent, 5, seeds)
        variants.append(
            {
                "reward_mode": reward_mode,
                "checkpoint_step": checkpoint["steps"],
                **summary,
            }
        )
        histories[reward_mode] = history
    return {
        "experiment": "lesson-19-full-observation-dqn",
        "training_seed": 42,
        "training_steps": 100_000,
        "evaluation_seeds": seeds,
        "variants": variants,
        "histories": histories,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DoorKey DQN rewards")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/doorkey-dqn"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/19-doorkey-full-dqn.json")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("docs/experiments/19-doorkey-full-dqn.csv")
    )
    args = parser.parse_args()
    report = build_report(args.runs_root, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report["variants"], args.csv_output)
    print(json.dumps(report["variants"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

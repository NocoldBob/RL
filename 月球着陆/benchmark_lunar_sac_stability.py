"""Aggregate the lesson 16 SAC capacity and entropy probes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_history(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation_at(history: dict[str, Any], step: int) -> dict[str, Any]:
    return next(row for row in history["evaluations"] if row["step"] == step)


def update_at_or_before(history: dict[str, Any], step: int) -> dict[str, Any]:
    return max(
        (row for row in history["updates"] if row["step"] <= step), key=lambda row: row["step"]
    )


def build_report(runs_root: Path) -> dict[str, Any]:
    specifications = (
        ("target_entropy_-2", runs_root / "lunar-sac-stable-128" / "seed-42", -2.0),
        ("target_entropy_-1", runs_root / "lunar-sac-entropy" / "seed-42", -1.0),
    )
    variants = []
    histories = {}
    for name, run_dir, target_entropy in specifications:
        history = load_history(run_dir / "history.json")
        histories[name] = history
        evaluation = evaluation_at(history, 50_000)
        update = update_at_or_before(history, 50_000)
        variants.append(
            {
                "variant": name,
                "target_entropy": target_entropy,
                "steps": 50_000,
                "average_return": evaluation["average_return"],
                "landing_rate": evaluation["landing_rate"],
                "crash_rate": evaluation["crash_rate"],
                "average_steps": evaluation["average_steps"],
                "alpha": update["alpha"],
                "q_loss": update["q_loss"],
                "q_disagreement": update["q_disagreement"],
                "log_probability": update["log_probability"],
            }
        )
    default_history = histories["target_entropy_-2"]
    return {
        "experiment": "lesson-16-lunar-sac-stability",
        "seed": 42,
        "network": "128x128 actor and twin Q networks",
        "update_frequency": 2,
        "fixed_evaluation_seeds": list(range(50_000, 50_020)),
        "variants_at_50k": variants,
        "default_evaluations": default_history["evaluations"],
        "default_updates": default_history["updates"],
        "conclusion": (
            "Larger networks improve return smoothly, but neither entropy target lands by 100k."
        ),
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    rows = report["variants_at_50k"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Lunar SAC stability probes")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/16-lunar-sac-stability.json")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("docs/experiments/16-lunar-sac-stability.csv")
    )
    args = parser.parse_args()
    report = build_report(args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report, args.csv_output)
    print(json.dumps(report["variants_at_50k"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

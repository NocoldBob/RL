"""Aggregate feed-forward and recurrent PPO memory experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(path: Path) -> dict[str, Any]:
    report = load(path)
    evaluations = report["evaluations"]
    best = max(evaluations, key=lambda row: row["success_rate"])
    return {
        "best_step": best["step"],
        "best_success_rate": best["success_rate"],
        "final_success_rate": evaluations[-1]["success_rate"],
        "final_average_return": evaluations[-1]["average_return"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate DoorKey memory experiments")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/doorkey-ppo"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/21-doorkey-memory.json")
    )
    args = parser.parse_args()
    variants = {
        "feedforward": args.runs_root / "feedforward-seed-42" / "history.json",
        "gru-naive-single-step": args.runs_root / "gru-seed-42" / "history.json",
        "gru-truncated-bptt": args.runs_root / "gru-bptt-seed-42" / "history.json",
    }
    report = {
        "experiment": "lesson-21-doorkey-memory",
        "variants": {name: summarize(path) for name, path in variants.items()},
        "histories": {name: load(path) for name, path in variants.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["variants"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

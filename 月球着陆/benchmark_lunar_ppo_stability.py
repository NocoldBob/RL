"""Aggregate the paired LunarLander PPO stability experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_history(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "history.json").read_text(encoding="utf-8"))


def summarize(seed: int, variant: str, history: dict[str, Any]) -> dict[str, Any]:
    evaluations = history["evaluations"]
    updates = history["updates"]
    best = max(evaluations, key=lambda row: (row["landing_rate"], row["average_return"]))
    final = evaluations[-1]
    return {
        "variant": variant,
        "seed": seed,
        "best_step": best["step"],
        "best_average_return": best["average_return"],
        "best_landing_rate": best["landing_rate"],
        "final_average_return": final["average_return"],
        "final_landing_rate": final["landing_rate"],
        "final_crash_rate": final["crash_rate"],
        "early_stop_count": sum(row["early_stopped"] for row in updates),
        "mean_approx_kl": sum(row["approx_kl"] for row in updates) / len(updates),
        "mean_clip_fraction": sum(row["clip_fraction"] for row in updates) / len(updates),
        "final_entropy": updates[-1]["entropy"],
        "final_explained_variance": updates[-1]["explained_variance"],
    }


def build_report(root: Path, seeds: list[int]) -> dict[str, Any]:
    variants = {
        "conservative": "lunar-ppo-stable",
        "stabilized": "lunar-ppo-stable-v2",
    }
    report: dict[str, Any] = {
        "experiment": "lesson-15-lunar-ppo-stability",
        "seeds": seeds,
        "fixed_evaluation_seeds": list(range(50_000, 50_020)),
        "variants": {},
    }
    for variant, directory in variants.items():
        runs = []
        histories = {}
        for seed in seeds:
            history = load_history(root / directory / f"seed-{seed}")
            histories[str(seed)] = history
            runs.append(summarize(seed, variant, history))
        report["variants"][variant] = {
            "runs": runs,
            "histories": histories,
            "success_rate": sum(row["final_landing_rate"] >= 0.6 for row in runs) / len(runs),
            "mean_final_landing_rate": sum(row["final_landing_rate"] for row in runs) / len(runs),
        }
    report["conclusion"] = {
        "screening_gate": "at least two of three seeds reach 60% landing rate at 200k steps",
        "passed": sum(
            row["final_landing_rate"] >= 0.6 for row in report["variants"]["stabilized"]["runs"]
        )
        >= 2,
    }
    return report


def write_csv(report: dict[str, Any], destination: Path) -> None:
    rows = [row for variant in report["variants"].values() for row in variant["runs"]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Lunar PPO stability runs")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 2026])
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/15-lunar-ppo-stability.json")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("docs/experiments/15-lunar-ppo-stability.csv")
    )
    args = parser.parse_args()
    report = build_report(args.runs_root, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report, args.csv_output)
    print(json.dumps(report["conclusion"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

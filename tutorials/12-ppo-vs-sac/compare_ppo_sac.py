"""Build the lesson 12 paired PPO/SAC comparison report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from continuous_ppo import ContinuousPPOAgent
from sac import SACAgent
from visualize_ppo_sac import generate_assets

METRICS = (
    "average_return",
    "success_rate",
    "average_steps",
    "average_success_steps",
    "average_action_cost",
)


def parameter_counts() -> dict[str, dict[str, int]]:
    ppo = ContinuousPPOAgent()
    sac = SACAgent()
    return {
        "ppo": {
            "training_parameters": sum(parameter.numel() for parameter in ppo.model.parameters()),
            "inference_parameters": sum(parameter.numel() for parameter in ppo.model.parameters()),
        },
        "sac": {
            "training_parameters": sum(
                parameter.numel()
                for module in (sac.actor, sac.q1, sac.q2, sac.target_q1, sac.target_q2)
                for parameter in module.parameters()
            )
            + 1,
            "inference_parameters": sum(parameter.numel() for parameter in sac.actor.parameters()),
        },
    }


def optimizer_workloads(ppo_report: dict[str, Any], sac_report: dict[str, Any]) -> dict[str, Any]:
    ppo_cycles = {}
    for seed, history in ppo_report["ppo"]["histories"].items():
        ppo_cycles[seed] = sum(
            10 * math.ceil(update["samples"] / 64) for update in history["updates"]
        )
    total_steps = int(sac_report["experiment"]["total_steps_per_seed"])
    learning_starts = 5_000
    return {
        "ppo": {
            "minibatch_update_cycles_by_seed": ppo_cycles,
            "optimizers_per_cycle": 1,
            "reuses_each_rollout_for_epochs": 10,
        },
        "sac": {
            "update_cycles_per_seed": total_steps - learning_starts,
            "optimizers_per_cycle": 3,
            "batch_size": 128,
        },
        "note": "Update counts describe different computations and are not wall-clock timings.",
    }


def build_comparison(ppo_report: dict[str, Any], sac_report: dict[str, Any]) -> dict[str, Any]:
    ppo_experiment = ppo_report["experiment"]
    sac_experiment = sac_report["experiment"]
    for name in ("training_seeds", "total_steps_per_seed", "evaluation_seeds"):
        if ppo_experiment[name] != sac_experiment[name]:
            raise ValueError(f"PPO and SAC reports do not share {name}")

    ppo_seeds = {row["training_seed"]: row for row in ppo_report["ppo"]["training_seeds"]}
    sac_seeds = {row["training_seed"]: row for row in sac_report["sac"]["training_seeds"]}
    paired_rows = []
    for seed in ppo_experiment["training_seeds"]:
        ppo_summary = ppo_seeds[seed]["summary"]
        sac_summary = sac_seeds[seed]["summary"]
        row: dict[str, Any] = {"training_seed": seed}
        for metric in METRICS:
            row[f"ppo_{metric}"] = ppo_summary[metric]
            row[f"sac_{metric}"] = sac_summary[metric]
            if ppo_summary[metric] is not None and sac_summary[metric] is not None:
                row[f"sac_minus_ppo_{metric}"] = sac_summary[metric] - ppo_summary[metric]
            else:
                row[f"sac_minus_ppo_{metric}"] = None
        paired_rows.append(row)

    aggregates = {
        "ppo": ppo_report["ppo"]["aggregate_across_training_seeds"],
        "sac": sac_report["sac"]["aggregate_across_training_seeds"],
    }
    reliable_threshold = 0.9
    return {
        "schema_version": 1,
        "environment": ppo_report["environment"],
        "fairness": {
            "training_seeds": ppo_experiment["training_seeds"],
            "environment_steps_per_seed": ppo_experiment["total_steps_per_seed"],
            "evaluation_seeds": ppo_experiment["evaluation_seeds"],
            "deterministic_evaluation": True,
            "reward_shaping": False,
            "teacher_demonstrations": False,
            "important_difference": (
                "SAC uses 5000 steps of temporally correlated random exploration."
            ),
        },
        "aggregate": aggregates,
        "paired_training_seeds": paired_rows,
        "reliable_training_seeds": {
            "threshold": reliable_threshold,
            "ppo": sum(row["ppo_success_rate"] >= reliable_threshold for row in paired_rows),
            "sac": sum(row["sac_success_rate"] >= reliable_threshold for row in paired_rows),
            "total": len(paired_rows),
        },
        "parameter_counts": parameter_counts(),
        "optimizer_workloads": optimizer_workloads(ppo_report, sac_report),
        "histories": {
            "ppo": ppo_report["ppo"]["histories"],
            "sac": sac_report["sac"]["histories"],
        },
        "sources": {
            "ppo": "docs/experiments/10-continuous-ppo.json",
            "sac": "docs/experiments/11-sac.json",
        },
    }


def write_report(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = report["paired_training_seeds"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare lesson 10 PPO and lesson 11 SAC")
    parser.add_argument(
        "--ppo-json", type=Path, default=Path("docs/experiments/10-continuous-ppo.json")
    )
    parser.add_argument("--sac-json", type=Path, default=Path("docs/experiments/11-sac.json"))
    parser.add_argument(
        "--output-json", type=Path, default=Path("docs/experiments/12-ppo-vs-sac.json")
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("docs/experiments/12-ppo-vs-sac.csv")
    )
    parser.add_argument("--assets-dir", type=Path, default=Path("docs/assets/csdn-12"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ppo_report = json.loads(args.ppo_json.read_text(encoding="utf-8"))
    sac_report = json.loads(args.sac_json.read_text(encoding="utf-8"))
    report = build_comparison(ppo_report, sac_report)
    write_report(report, args.output_json, args.output_csv)
    generate_assets(report, args.assets_dir)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

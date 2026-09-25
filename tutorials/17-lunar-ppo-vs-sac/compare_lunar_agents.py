"""Compare LunarLander PPO and SAC under explicit evaluation scopes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic
from lunar_agents import LunarPPOAgent, LunarSACAgent
from lunar_baselines import ENV_ID, engine_power
from lunar_evaluation import evaluate_lunar_agent

SCENARIOS = {
    "left_high_drifting_out": [-0.8, 0.8, -0.25, -0.3, 0.15, 0.1, 0.0, 0.0],
    "right_tilted_fast": [0.6, 0.5, 0.3, -0.45, -0.25, -0.2, 0.0, 0.0],
    "near_ground_fast_descent": [0.1, 0.15, 0.05, -0.5, 0.05, 0.0, 0.0, 0.0],
    "one_leg_contact": [0.05, 0.05, 0.0, -0.1, 0.1, 0.2, 1.0, 0.0],
    "both_legs_contact": [0.0, 0.0, 0.0, -0.05, 0.02, 0.0, 1.0, 1.0],
}


def load_history(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation_at(history: dict[str, Any], step: int) -> dict[str, Any]:
    return next(row for row in history["evaluations"] if row["step"] == step)


def inspect_scenarios(ppo: LunarPPOAgent, sac: LunarSACAgent) -> list[dict[str, Any]]:
    env = gym.make(ENV_ID)
    rows = []
    try:
        for name, values in SCENARIOS.items():
            observation = np.asarray(values, dtype=np.float32)
            actions = {
                "heuristic": np.asarray(heuristic(env, observation), dtype=np.float32),
                "ppo": ppo.predict(observation),
                "sac": sac.predict(observation),
            }
            for policy, action in actions.items():
                main_power, side_power = engine_power(action)
                rows.append(
                    {
                        "scenario": name,
                        "policy": policy,
                        "main_action": float(action[0]),
                        "side_action": float(action[1]),
                        "main_power": main_power,
                        "side_power": side_power,
                    }
                )
    finally:
        env.close()
    return rows


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    ppo = LunarPPOAgent()
    ppo.load_checkpoint(args.ppo_checkpoint)
    sac = LunarSACAgent()
    sac.load_checkpoint(args.sac_checkpoint)
    ppo_history = load_history(args.ppo_history)
    sac_history = load_history(args.sac_history)
    common_step = 100_000
    common = {
        "ppo": evaluation_at(ppo_history, common_step),
        "sac": evaluation_at(sac_history, common_step),
    }
    deployment_seeds = list(range(60_000, 60_000 + args.deployment_episodes))
    ppo_summary, ppo_rows, _ = evaluate_lunar_agent(ppo, deployment_seeds)
    sac_summary, sac_rows, _ = evaluate_lunar_agent(sac, deployment_seeds)
    return {
        "experiment": "lesson-17-lunar-ppo-sac-comparison",
        "common_budget": {
            "seed": 42,
            "steps": common_step,
            "evaluation_seeds": list(range(50_000, 50_020)),
            "results": common,
        },
        "best_checkpoint_evaluation": {
            "notice": "Deployment comparison; checkpoints have different training budgets.",
            "evaluation_seeds": deployment_seeds,
            "ppo": {"summary": ppo_summary, "episodes": ppo_rows},
            "sac": {"summary": sac_summary, "episodes": sac_rows},
        },
        "fixed_scenarios": inspect_scenarios(ppo, sac),
    }


def write_scenario_csv(report: dict[str, Any], path: Path) -> None:
    rows = report["fixed_scenarios"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Lunar PPO and SAC")
    parser.add_argument(
        "--ppo-history", type=Path, default=Path("runs/lunar-ppo-stable-v2/seed-42/history.json")
    )
    parser.add_argument(
        "--sac-history", type=Path, default=Path("runs/lunar-sac-stable-128/seed-42/history.json")
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        default=Path("runs/lunar-ppo-stable-v2/seed-42/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--sac-checkpoint",
        type=Path,
        default=Path("runs/lunar-sac-stable-128/seed-42/checkpoints/best.pt"),
    )
    parser.add_argument("--deployment-episodes", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/17-lunar-ppo-vs-sac.json")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("docs/experiments/17-lunar-fixed-scenarios.csv")
    )
    args = parser.parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_scenario_csv(report, args.csv_output)
    print(json.dumps(report["common_budget"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

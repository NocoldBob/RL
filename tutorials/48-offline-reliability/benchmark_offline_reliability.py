"""Audit offline policies under unseen seeds, observation noise, and sticky actions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from acrobot_env import evaluate_policy
from benchmark_offline_agents import load_agent
from online_dqn import DQNAgent


def main() -> None:
    torch.set_num_threads(1)
    specifications = {
        "bc": ("offline", Path("runs/offline/bc-mixed/checkpoints/best.pt")),
        "cql": ("offline", Path("runs/offline/cql-mixed-alpha-1/checkpoints/best.pt")),
        "iql": ("offline", Path("runs/offline/iql-mixed/checkpoints/best.pt")),
        "finetuned_mixed": ("online", Path("runs/offline-finetune/mixed_replay/best.pt")),
    }
    results = {}
    for name, (kind, path) in specifications.items():
        if kind == "offline":
            agent, step, _ = load_agent(path)
        else:
            agent = DQNAgent()
            step = agent.load(path)
        conditions = {
            "clean": {},
            "observation_noise": {"observation_noise": 0.05},
            "sticky_actions": {"action_sticky_probability": 0.2},
        }
        condition_results = {}
        for condition, kwargs in conditions.items():
            report = evaluate_policy(agent.act, list(range(95_000, 95_100)), **kwargs)
            steps = np.asarray([row["steps"] for row in report["episodes"]])
            condition_results[condition] = {
                **report,
                "p90_steps": float(np.percentile(steps, 90)),
                "worst_steps": int(steps.max()),
            }
        results[name] = {"checkpoint_step": step, "conditions": condition_results}
    report = {"experiment": "48-offline-reliability", "results": results}
    path = Path("docs/experiments/48-offline-reliability.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    condition: row["summary"]["average_steps"]
                    for condition, row in payload["conditions"].items()
                }
                for name, payload in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

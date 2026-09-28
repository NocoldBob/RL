"""Compare IS, WIS, and FQE estimates with hidden online evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from benchmark_offline_agents import load_agent
from dataset import OfflineDataset
from off_policy_evaluation import FittedQEvaluator, importance_sampling


def true_discounted_return(agent, episodes: int = 100, gamma: float = 0.99) -> dict[str, float]:
    environment = gym.make("Acrobot-v1")
    returns = []
    try:
        for episode in range(episodes):
            observation, _ = environment.reset(seed=80_000 + episode)
            done = False
            total = 0.0
            discount = 1.0
            while not done:
                observation, reward, terminated, truncated, _ = environment.step(
                    agent.act(observation)
                )
                total += discount * float(reward)
                discount *= gamma
                done = terminated or truncated
            returns.append(total)
    finally:
        environment.close()
    return {"mean": float(np.mean(returns)), "std": float(np.std(returns))}


def main() -> None:
    torch.set_num_threads(1)
    dataset = OfflineDataset.load(Path("datasets/acrobot/mixed.npz"))
    specifications = {
        "bc": Path("runs/offline/bc-mixed/checkpoints/best.pt"),
        "cql": Path("runs/offline/cql-mixed-alpha-1/checkpoints/best.pt"),
        "iql": Path("runs/offline/iql-mixed/checkpoints/best.pt"),
    }
    results: dict[str, Any] = {}
    for name, checkpoint in specifications.items():
        agent, step, _ = load_agent(checkpoint)
        fqe = FittedQEvaluator()
        history = fqe.train(dataset, agent)
        results[name] = {
            "checkpoint_step": step,
            "importance_sampling": importance_sampling(dataset, agent),
            "fqe_estimate": fqe.estimate(dataset, agent),
            "fqe_history": history,
            "true_discounted_return": true_discounted_return(agent),
        }
    report = {"experiment": "46-off-policy-evaluation", "results": results}
    path = Path("docs/experiments/46-off-policy-evaluation.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    "IS": row["importance_sampling"]["ordinary_is"],
                    "WIS": row["importance_sampling"]["weighted_is"],
                    "FQE": row["fqe_estimate"],
                    "true": row["true_discounted_return"]["mean"],
                }
                for name, row in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

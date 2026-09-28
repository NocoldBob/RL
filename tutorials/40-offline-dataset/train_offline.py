"""Train an offline agent without importing or stepping an environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dataset import OfflineDataset
from offline_agents import OfflineAgent


@dataclass(slots=True)
class OfflineConfig:
    mode: str = "bc"
    dataset: Path = Path("datasets/acrobot/expert.npz")
    gradient_steps: int = 20_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    cql_alpha: float = 1.0
    expectile: float = 0.7
    advantage_beta: float = 3.0
    validation_interval: int = 1_000
    target_interval: int = 250
    seed: int = 42
    output_dir: Path = Path("runs/offline/default")

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset"] = str(self.dataset)
        data["output_dir"] = str(self.output_dir)
        return data


def train(config: OfflineConfig) -> dict[str, Any]:
    torch.set_num_threads(1)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = OfflineDataset.load(config.dataset)
    training, validation = dataset.split_by_episode()
    agent = OfflineAgent(
        config.mode,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        cql_alpha=config.cql_alpha,
        expectile=config.expectile,
        advantage_beta=config.advantage_beta,
    )
    history: dict[str, list[dict[str, float]]] = {"updates": [], "validations": []}
    best_validation = float("inf")
    validation_indices = np.arange(min(4096, len(validation)))
    validation_batch = validation.batch(validation_indices, agent.device)
    for step in range(1, config.gradient_steps + 1):
        metrics = agent.update(training.sample(config.batch_size, agent.device))
        if step % 100 == 0:
            history["updates"].append({"step": float(step), **metrics})
        if step % config.target_interval == 0:
            agent.sync_target()
        if step % config.validation_interval == 0 or step == config.gradient_steps:
            validation_metrics = agent.validation_metrics(validation_batch)
            history["validations"].append({"step": float(step), **validation_metrics})
            objective = validation_metrics["validation_loss"]
            if objective < best_validation:
                best_validation = objective
                agent.save(config.output_dir / "checkpoints" / "best.pt", step, config.payload())
            (config.output_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"{config.mode} step={step} validation={objective:.4f}")
    agent.save(
        config.output_dir / "checkpoints" / "latest.pt", config.gradient_steps, config.payload()
    )
    summary = {
        "algorithm": config.mode,
        "best_validation_loss": best_validation,
        "training_transitions": len(training),
        "validation_transitions": len(validation),
        **config.payload(),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bc", "dqn", "cql", "iql"), required=True)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/acrobot/expert.npz"))
    parser.add_argument("--gradient-steps", type=int, default=20_000)
    parser.add_argument("--cql-alpha", type=float, default=1.0)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--advantage-beta", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    label = f"{args.mode}-{args.dataset.stem}"
    if args.mode == "cql":
        label += f"-alpha-{args.cql_alpha:g}"
    output = args.output_dir or Path("runs/offline") / label
    result = train(
        OfflineConfig(
            mode=args.mode,
            dataset=args.dataset,
            gradient_steps=args.gradient_steps,
            cql_alpha=args.cql_alpha,
            expectile=args.expectile,
            advantage_beta=args.advantage_beta,
            seed=args.seed,
            output_dir=output,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

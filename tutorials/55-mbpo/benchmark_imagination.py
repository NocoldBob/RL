"""Compare Dyna replay ratios and MBPO rollout horizons under one real-data budget."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from dynamics_model import DynamicsDataset, load_model, write_json
from model_based_sac import train_with_imagination
from pendulum_env import evaluate_controller


def train_group(
    configurations: dict[str, tuple[float, int]],
    real: DynamicsDataset,
    *,
    prefix: str,
    evaluation_seeds: list[int],
) -> dict:
    model, _ = load_model(Path("runs/model-based/dynamics/model-0.pt"))
    results = {}
    for index, (name, (fraction, horizon)) in enumerate(configurations.items()):
        agent, training = train_with_imagination(
            real,
            model,
            imagined_fraction=fraction,
            rollout_horizon=horizon,
            seed=540 + index,
        )
        evaluation = evaluate_controller(agent.act, evaluation_seeds)
        checkpoint = Path(f"runs/model-based/{prefix}-{name}.pt")
        agent.save(checkpoint, {"name": name, **training})
        results[name] = {
            "training": training,
            "evaluation": evaluation,
            "checkpoint": str(checkpoint),
        }
        print(f"{prefix} {name}: {evaluation['summary']['average_return']:.1f}")
    return results


def main() -> None:
    torch.set_num_threads(1)
    dataset = DynamicsDataset.load(Path("datasets/pendulum-model-based/transitions.npz"))
    real = dataset.select(dataset.episode_id < 20)
    dyna = train_group(
        {
            "real_only": (0.0, 1),
            "imagined_25": (0.25, 1),
            "imagined_50": (0.50, 1),
            "imagined_75": (0.75, 1),
        },
        real,
        prefix="dyna",
        evaluation_seeds=list(range(73_000, 73_020)),
    )
    write_json(
        Path("docs/experiments/54-dyna.json"),
        {"experiment": "54-dyna", "real_transitions": len(real), "results": dyna},
    )
    mbpo = train_group(
        {
            "horizon_1": (0.5, 1),
            "horizon_3": (0.5, 3),
            "horizon_5": (0.5, 5),
            "horizon_10": (0.5, 10),
        },
        real,
        prefix="mbpo",
        evaluation_seeds=list(range(74_000, 74_020)),
    )
    write_json(
        Path("docs/experiments/55-mbpo.json"),
        {"experiment": "55-mbpo", "real_transitions": len(real), "results": mbpo},
    )
    print(
        json.dumps(
            {
                "dyna": {k: v["evaluation"]["summary"]["average_return"] for k, v in dyna.items()},
                "mbpo": {k: v["evaluation"]["summary"]["average_return"] for k, v in mbpo.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

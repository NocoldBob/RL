"""Benchmark random-shooting and CEM planning with learned and oracle models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from dynamics_model import MPCController, OracleDynamics, load_model, write_json
from pendulum_env import evaluate_controller, heuristic_action


class RandomController:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return self.rng.uniform(-2.0, 2.0, size=1).astype(np.float32)


def main() -> None:
    torch.set_num_threads(1)
    model, _ = load_model(Path("runs/model-based/dynamics/model-0.pt"))
    seeds = list(range(71_000, 71_010))
    lesson_51 = {
        "random": evaluate_controller(RandomController(51), seeds),
        "heuristic": evaluate_controller(heuristic_action, seeds),
        "oracle_mpc": evaluate_controller(
            MPCController(OracleDynamics(), method="random", horizon=20, candidates=256, seed=51),
            seeds,
        ),
        "learned_mpc": evaluate_controller(
            MPCController(model, method="random", horizon=20, candidates=256, seed=51), seeds
        ),
    }
    write_json(
        Path("docs/experiments/51-random-shooting-mpc.json"),
        {"experiment": "51-random-shooting-mpc", "results": lesson_51},
    )
    lesson_52 = {}
    configurations = {
        "random_256": dict(method="random", candidates=256, iterations=1),
        "cem_64": dict(method="cem", candidates=64, iterations=3),
        "cem_128": dict(method="cem", candidates=128, iterations=3),
        "cem_256": dict(method="cem", candidates=256, iterations=3),
    }
    for name, config in configurations.items():
        controller = MPCController(model, horizon=20, seed=52, **config)
        lesson_52[name] = evaluate_controller(controller, seeds)
    write_json(
        Path("docs/experiments/52-cem-planning.json"),
        {"experiment": "52-cem-planning", "results": lesson_52},
    )
    print(
        json.dumps(
            {
                "lesson_51": {k: v["summary"]["average_return"] for k, v in lesson_51.items()},
                "lesson_52": {k: v["summary"]["average_return"] for k, v in lesson_52.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

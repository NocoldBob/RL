"""Expose model exploitation in long-horizon offline planning."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from dynamics_model import ModelEnsemble, MPCController, load_model, write_json
from pendulum_env import evaluate_controller, heuristic_action


def main() -> None:
    torch.set_num_threads(1)
    models = [
        load_model(Path(f"runs/model-based/dynamics/model-{index}.pt"))[0] for index in range(5)
    ]
    ensemble = ModelEnsemble(models)
    seeds = list(range(75_000, 75_005))
    controllers = {
        "behavior_reference": heuristic_action,
        "naive_long_horizon": MPCController(
            models[0], method="cem", horizon=30, candidates=64, iterations=2, seed=56
        ),
        "ensemble_long_horizon": MPCController(
            ensemble, method="cem", horizon=30, candidates=64, iterations=2, seed=56
        ),
        "uncertainty_penalty": MPCController(
            ensemble,
            method="cem",
            horizon=30,
            candidates=64,
            iterations=2,
            uncertainty_penalty=1.0,
            seed=56,
        ),
    }
    results = {
        name: evaluate_controller(controller, seeds) for name, controller in controllers.items()
    }
    report = {"experiment": "56-offline-model-planning", "results": results}
    write_json(Path("docs/experiments/56-offline-model-planning.json"), report)
    print(json.dumps({k: v["summary"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()

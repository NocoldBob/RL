"""Compare single-model, ensemble, and uncertainty-penalized planning."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from dynamics_model import ModelEnsemble, MPCController, load_model, write_json
from pendulum_env import evaluate_controller


def main() -> None:
    torch.set_num_threads(1)
    models = [
        load_model(Path(f"runs/model-based/dynamics/model-{index}.pt"))[0] for index in range(5)
    ]
    ensemble = ModelEnsemble(models)
    seeds = list(range(72_000, 72_005))
    controllers = {
        "single": MPCController(
            models[0], method="cem", horizon=15, candidates=64, iterations=2, seed=53
        ),
        "ensemble": MPCController(
            ensemble, method="cem", horizon=15, candidates=64, iterations=2, seed=53
        ),
        "pessimistic_0.5": MPCController(
            ensemble,
            method="cem",
            horizon=15,
            candidates=64,
            iterations=2,
            uncertainty_penalty=0.5,
            seed=53,
        ),
        "pessimistic_2.0": MPCController(
            ensemble,
            method="cem",
            horizon=15,
            candidates=64,
            iterations=2,
            uncertainty_penalty=2.0,
            seed=53,
        ),
    }
    results = {
        name: evaluate_controller(controller, seeds) for name, controller in controllers.items()
    }
    report = {"experiment": "53-ensemble-uncertainty", "results": results}
    write_json(Path("docs/experiments/53-ensemble-uncertainty.json"), report)
    print(json.dumps({k: v["summary"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()

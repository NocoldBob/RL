"""Audit planning and MBPO policies under dynamics and sensing shifts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from dynamics_model import ModelEnsemble, MPCController, load_model, write_json
from model_based_sac import SimpleSAC
from pendulum_env import evaluate_controller, heuristic_action


def main() -> None:
    torch.set_num_threads(1)
    models = [
        load_model(Path(f"runs/model-based/dynamics/model-{index}.pt"))[0] for index in range(5)
    ]
    planner = MPCController(
        ModelEnsemble(models),
        method="cem",
        horizon=15,
        candidates=48,
        iterations=2,
        uncertainty_penalty=0.5,
        seed=57,
    )
    mbpo, metadata = SimpleSAC.load(Path("runs/model-based/mbpo-horizon_1.pt"))
    controllers = {"heuristic": heuristic_action, "ensemble_mpc": planner, "mbpo": mbpo.act}
    conditions = {
        "clean": {},
        "heavy": {"mass": 1.25},
        "short_link": {"length": 0.8},
        "observation_noise": {"observation_noise": 0.05},
        "action_delay": {"action_delay": 2},
    }
    seeds = list(range(76_000, 76_005))
    results = {
        name: {
            condition: evaluate_controller(controller, seeds, **kwargs)
            for condition, kwargs in conditions.items()
        }
        for name, controller in controllers.items()
    }
    report = {"experiment": "57-model-reliability", "mbpo_metadata": metadata, "results": results}
    write_json(Path("docs/experiments/57-model-reliability.json"), report)
    print(
        json.dumps(
            {
                name: {
                    condition: row["summary"]["average_return"] for condition, row in rows.items()
                }
                for name, rows in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Train five independently bootstrapped dynamics models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from dynamics_model import DynamicsDataset, save_model, train_model, write_json


def main() -> None:
    torch.set_num_threads(1)
    dataset = DynamicsDataset.load(Path("datasets/pendulum-model-based/transitions.npz"))
    training, validation = dataset.split_by_episode()
    results = []
    for index, seed in enumerate((11, 22, 33, 44, 55)):
        rng = np.random.default_rng(seed)
        bootstrap = training.select(rng.integers(0, len(training), size=len(training)))
        model, report = train_model(bootstrap, validation, seed=seed)
        path = Path(f"runs/model-based/dynamics/model-{index}.pt")
        save_model(model, path, report)
        results.append({"model": index, "checkpoint": str(path), **report})
        print(f"model={index} validation={report['best_validation_loss']:.5f}")
    payload = {
        "experiment": "49-dynamics-model",
        "training_transitions": len(training),
        "validation_transitions": len(validation),
        "models": results,
    }
    write_json(Path("docs/experiments/49-dynamics-model.json"), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

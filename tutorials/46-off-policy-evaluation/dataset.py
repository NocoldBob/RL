"""Versioned NumPy transition datasets for offline Acrobot experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

FIELDS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminated",
    "truncated",
    "episode_id",
    "behavior_probability",
    "source",
)


@dataclass(slots=True)
class OfflineDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_id: np.ndarray
    behavior_probability: np.ndarray
    source: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    @classmethod
    def load(cls, path: Path) -> OfflineDataset:
        with np.load(path) as payload:
            return cls(**{field: payload[field] for field in FIELDS})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{field: getattr(self, field) for field in FIELDS})

    def select(self, indices: np.ndarray) -> OfflineDataset:
        return OfflineDataset(**{field: getattr(self, field)[indices] for field in FIELDS})

    def split_by_episode(
        self, validation_fraction: float = 0.2
    ) -> tuple[OfflineDataset, OfflineDataset]:
        episodes = np.unique(self.episode_id)
        boundary = max(1, int(len(episodes) * (1 - validation_fraction)))
        training_episodes = episodes[:boundary]
        training_mask = np.isin(self.episode_id, training_episodes)
        return self.select(np.flatnonzero(training_mask)), self.select(
            np.flatnonzero(~training_mask)
        )

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(len(self), size=batch_size)
        return self.batch(indices, device)

    def batch(self, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "observations": torch.as_tensor(self.observations[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_observations": torch.as_tensor(self.next_observations[indices], device=device),
            "terminated": torch.as_tensor(
                self.terminated[indices], device=device, dtype=torch.float32
            ),
            "truncated": torch.as_tensor(
                self.truncated[indices], device=device, dtype=torch.float32
            ),
            "behavior_probability": torch.as_tensor(
                self.behavior_probability[indices], device=device
            ),
        }

    def summary(self) -> dict[str, Any]:
        episodes = np.unique(self.episode_id)
        returns = np.asarray(
            [self.rewards[self.episode_id == episode].sum() for episode in episodes]
        )
        successes = np.asarray(
            [self.terminated[self.episode_id == episode].any() for episode in episodes]
        )
        rounded_states = np.round(self.observations, decimals=1)
        return {
            "transitions": len(self),
            "episodes": len(episodes),
            "success_rate": float(successes.mean()),
            "average_return": float(returns.mean()),
            "average_episode_length": float(len(self) / len(episodes)),
            "action_distribution": [float(np.mean(self.actions == action)) for action in range(3)],
            "unique_rounded_states": len(np.unique(rounded_states, axis=0)),
            "observation_mean": self.observations.mean(axis=0).tolist(),
            "observation_std": self.observations.std(axis=0).tolist(),
        }


def concatenate(datasets: list[OfflineDataset]) -> OfflineDataset:
    offset = 0
    arrays: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
    for dataset in datasets:
        for field in FIELDS:
            values = getattr(dataset, field)
            if field == "episode_id":
                values = values + offset
            arrays[field].append(values)
        offset += int(dataset.episode_id.max()) + 1
    return OfflineDataset(**{field: np.concatenate(values) for field, values in arrays.items()})


def write_manifest(path: Path, datasets: dict[str, OfflineDataset]) -> None:
    payload = {
        "format_version": 1,
        "environment": "Acrobot-v1",
        "fields": list(FIELDS),
        "datasets": {name: dataset.summary() for name, dataset in datasets.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

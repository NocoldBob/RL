"""Versioned Pendulum datasets, learned dynamics, ensembles, and MPC planners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass(slots=True)
class DynamicsDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    episode_id: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    @classmethod
    def load(cls, path: Path) -> DynamicsDataset:
        with np.load(path) as payload:
            return cls(**{name: payload[name] for name in cls.__annotations__})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{name: getattr(self, name) for name in self.__annotations__})

    def split_by_episode(
        self, validation_fraction: float = 0.2
    ) -> tuple[DynamicsDataset, DynamicsDataset]:
        episodes = np.unique(self.episode_id)
        boundary = max(1, int(len(episodes) * (1 - validation_fraction)))
        mask = np.isin(self.episode_id, episodes[:boundary])
        return self.select(mask), self.select(~mask)

    def select(self, mask: np.ndarray) -> DynamicsDataset:
        return DynamicsDataset(**{name: getattr(self, name)[mask] for name in self.__annotations__})

    def summary(self) -> dict[str, Any]:
        return {
            "transitions": len(self),
            "episodes": len(np.unique(self.episode_id)),
            "average_reward": float(self.rewards.mean()),
            "observation_mean": self.observations.mean(axis=0).tolist(),
            "observation_std": self.observations.std(axis=0).tolist(),
            "action_mean": float(self.actions.mean()),
            "action_std": float(self.actions.std()),
        }


class DynamicsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
        )
        self.register_buffer("input_mean", torch.zeros(4))
        self.register_buffer("input_std", torch.ones(4))
        self.register_buffer("target_mean", torch.zeros(4))
        self.register_buffer("target_std", torch.ones(4))

    def set_statistics(self, dataset: DynamicsDataset) -> None:
        inputs = np.concatenate([dataset.observations, dataset.actions], axis=1)
        targets = np.concatenate(
            [dataset.next_observations - dataset.observations, dataset.rewards[:, None]], axis=1
        )
        self.input_mean.copy_(torch.as_tensor(inputs.mean(axis=0)))
        self.input_std.copy_(torch.as_tensor(inputs.std(axis=0)).clamp_min(1e-5))
        self.target_mean.copy_(torch.as_tensor(targets.mean(axis=0)))
        self.target_std.copy_(torch.as_tensor(targets.std(axis=0)).clamp_min(1e-5))

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        normalized = (inputs - self.input_mean) / self.input_std
        return self.network(normalized) * self.target_std + self.target_mean

    @torch.no_grad()
    def predict(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self(observations, actions)
        next_observations = observations + outputs[:, :3]
        norm = next_observations[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-6)
        next_observations = torch.cat(
            [next_observations[:, :2] / norm, next_observations[:, 2:]], dim=1
        )
        return next_observations, outputs[:, 3]


def tensors(dataset: DynamicsDataset, indices: np.ndarray) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.as_tensor(value[indices], dtype=torch.float32)
        for value in (
            dataset.observations,
            dataset.actions,
            dataset.rewards,
            dataset.next_observations,
        )
    )


def train_model(
    training: DynamicsDataset,
    validation: DynamicsDataset,
    *,
    seed: int,
    gradient_steps: int = 3_000,
    batch_size: int = 256,
) -> tuple[DynamicsModel, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DynamicsModel()
    model.set_statistics(training)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    history = []
    best_loss = float("inf")
    best_state = None
    validation_indices = np.arange(len(validation))
    val_obs, val_actions, val_rewards, val_next = tensors(validation, validation_indices)
    for step in range(1, gradient_steps + 1):
        indices = rng.integers(0, len(training), size=batch_size)
        observations, actions, rewards, next_observations = tensors(training, indices)
        output = model(observations, actions)
        target = torch.cat([next_observations - observations, rewards[:, None]], dim=1)
        normalized_error = (output - target) / model.target_std
        loss = normalized_error.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 250 == 0:
            with torch.no_grad():
                predicted = model(val_obs, val_actions)
                target = torch.cat([val_next - val_obs, val_rewards[:, None]], dim=1)
                validation_loss = float(((predicted - target) / model.target_std).square().mean())
                state_rmse = float((predicted[:, :3] - target[:, :3]).square().mean().sqrt())
                reward_rmse = float((predicted[:, 3] - target[:, 3]).square().mean().sqrt())
            history.append(
                {
                    "step": step,
                    "validation_loss": validation_loss,
                    "state_rmse": state_rmse,
                    "reward_rmse": reward_rmse,
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_validation_loss": best_loss, "history": history}


def save_model(model: DynamicsModel, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_model(path: Path) -> tuple[DynamicsModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DynamicsModel()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metadata"]


class ModelEnsemble:
    def __init__(self, models: list[DynamicsModel]) -> None:
        self.models = models

    @torch.no_grad()
    def predict(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = [model.predict(observations, actions) for model in self.models]
        states = torch.stack([item[0] for item in predictions])
        rewards = torch.stack([item[1] for item in predictions])
        uncertainty = states.std(dim=0).norm(dim=1) + rewards.std(dim=0)
        mean_states = states.mean(dim=0)
        norm = mean_states[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-6)
        mean_states = torch.cat([mean_states[:, :2] / norm, mean_states[:, 2:]], dim=1)
        return mean_states, rewards.mean(dim=0), uncertainty


class OracleDynamics:
    """Vectorized Pendulum equations used as an upper-bound planning model."""

    @torch.no_grad()
    def predict(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = torch.atan2(observations[:, 1], observations[:, 0])
        velocity = observations[:, 2]
        torque = actions[:, 0].clamp(-2.0, 2.0)
        rewards = -(theta.square() + 0.1 * velocity.square() + 0.001 * torque.square())
        next_velocity = velocity + (15.0 * torch.sin(theta) + 3.0 * torque) * 0.05
        next_velocity = next_velocity.clamp(-8.0, 8.0)
        next_theta = theta + next_velocity * 0.05
        next_observations = torch.stack(
            [torch.cos(next_theta), torch.sin(next_theta), next_velocity], dim=1
        )
        return next_observations, rewards


class MPCController:
    def __init__(
        self,
        model: DynamicsModel | ModelEnsemble | OracleDynamics,
        *,
        method: str = "random",
        horizon: int = 20,
        candidates: int = 256,
        iterations: int = 4,
        uncertainty_penalty: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.method = method
        self.horizon = horizon
        self.candidates = candidates
        self.iterations = iterations
        self.uncertainty_penalty = uncertainty_penalty
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def _returns(self, observation: np.ndarray, sequences: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(observation, dtype=torch.float32).repeat(len(sequences), 1)
        returns = torch.zeros(len(sequences))
        for step in range(self.horizon):
            actions = sequences[:, step : step + 1]
            if isinstance(self.model, ModelEnsemble):
                states, rewards, uncertainty = self.model.predict(states, actions)
                rewards = rewards - self.uncertainty_penalty * uncertainty
            else:
                states, rewards = self.model.predict(states, actions)
            returns += rewards
        return returns

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self.method == "random":
            sequences = torch.empty(self.candidates, self.horizon).uniform_(
                -2.0, 2.0, generator=self.generator
            )
            best = sequences[self._returns(observation, sequences).argmax(), 0]
            return np.asarray([float(best)], dtype=np.float32)
        mean = torch.zeros(self.horizon)
        std = torch.full((self.horizon,), 2.0)
        for _ in range(self.iterations):
            noise = torch.randn(self.candidates, self.horizon, generator=self.generator)
            sequences = (mean + std * noise).clamp(-2.0, 2.0)
            scores = self._returns(observation, sequences)
            elite = sequences[scores.topk(max(8, self.candidates // 10)).indices]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0).clamp_min(0.1)
        return np.asarray([float(mean[0])], dtype=np.float32)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

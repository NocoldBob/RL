"""Small online Double DQN used only to create offline datasets."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from acrobot_env import ACTION_COUNT, OBSERVATION_SIZE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_COUNT),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.observations = np.empty((capacity, OBSERVATION_SIZE), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def append(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.terminated[index] = terminated
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(self.size, size=batch_size)
        return tuple(
            torch.as_tensor(array[indices], device=device)
            for array in (
                self.observations,
                self.actions,
                self.rewards,
                self.next_observations,
                self.terminated,
            )
        )


@dataclass(slots=True)
class OnlineConfig:
    total_steps: int = 80_000
    warmup_steps: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 5e-4
    target_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 50_000
    eval_interval: int = 10_000
    eval_episodes: int = 20
    seed: int = 42
    output_dir: Path = Path("runs/acrobot-online-dqn/seed-42")

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


class DQNAgent:
    def __init__(self, learning_rate: float = 5e-4, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.online = QNetwork().to(self.device)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

    @torch.no_grad()
    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(ACTION_COUNT))
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        return int(self.online(tensor).argmax(dim=1).item())

    def update(self, batch: tuple[torch.Tensor, ...], gamma: float) -> dict[str, float]:
        observations, actions, rewards, next_observations, terminated = batch
        predicted = self.online(observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_observations).argmax(dim=1, keepdim=True)
            next_values = self.target(next_observations).gather(1, next_actions).squeeze(1)
            targets = rewards + gamma * (1 - terminated) * next_values
        loss = F.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return {"loss": float(loss.item()), "q_mean": float(predicted.mean().item())}

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: Path, step: int, config: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.online.state_dict(), "step": step, "config": config}, path)

    def load(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(payload["model"])
        self.target.load_state_dict(payload["model"])
        return int(payload["step"])

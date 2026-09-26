"""CNN Double-DQN components for discretized CarRacing."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_encoder import NatureEncoder


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class VisualQNetwork(nn.Module):
    def __init__(self, channels: int, action_count: int, image_size: int = 64) -> None:
        super().__init__()
        self.encoder = NatureEncoder(channels, image_size)
        self.head = nn.Linear(256, action_count)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(observations))


class VisualReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple[int, int, int]) -> None:
        self.capacity = capacity
        self.states = np.empty((capacity, *observation_shape), dtype=np.uint8)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.done = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _compress(state: np.ndarray) -> np.ndarray:
        return np.clip(state * 255.0, 0, 255).astype(np.uint8)

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.states[index] = self._compress(state)
        self.next_states[index] = self._compress(next_state)
        self.actions[index] = action
        self.rewards[index] = reward
        self.done[index] = done
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        states = torch.as_tensor(self.states[indices], device=device, dtype=torch.float32) / 255.0
        next_states = (
            torch.as_tensor(self.next_states[indices], device=device, dtype=torch.float32) / 255.0
        )
        return (
            states,
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            next_states,
            torch.as_tensor(self.done[indices], device=device),
        )


@dataclass(slots=True)
class DQNConfig:
    frame_stack: int = 4
    image_size: int = 64
    total_steps: int = 80_000
    replay_capacity: int = 30_000
    warmup_steps: int = 2_000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-4
    train_interval: int = 4
    target_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 60_000
    eval_interval: int = 10_000
    eval_episodes: int = 5
    seed: int = 42
    output_dir: Path = Path("runs/carracing-dqn/default")

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


class VisualDQNAgent:
    def __init__(
        self,
        channels: int,
        action_count: int,
        *,
        image_size: int = 64,
        learning_rate: float = 1e-4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.online = VisualQNetwork(channels, action_count, image_size).to(self.device)
        self.target = VisualQNetwork(channels, action_count, image_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.action_count = action_count

    @torch.no_grad()
    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(self.action_count))
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        return int(self.online(tensor).argmax(dim=1).item())

    def update(self, batch: tuple[torch.Tensor, ...], gamma: float) -> dict[str, float]:
        states, actions, rewards, next_states, done = batch
        predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + gamma * (1.0 - done) * next_values
        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return {"loss": float(loss.item()), "q_mean": float(predicted.mean().item())}

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: Path, *, steps: int, config: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.online.state_dict(), "steps": steps, "config": config}, path)

    def load(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(payload["model"])
        self.target.load_state_dict(payload["model"])
        return int(payload.get("steps", 0))

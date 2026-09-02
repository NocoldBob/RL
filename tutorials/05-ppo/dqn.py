"""Small DQN agent and replay buffer for the Snake lesson."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """A seeded replay buffer that owns copies of environment observations."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.transitions: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.transitions)

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.transitions.append(
            (state.copy(), int(action), float(reward), next_state.copy(), bool(done))
        )

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.transitions):
            raise ValueError("batch_size exceeds the number of stored transitions")
        samples = self.random.sample(list(self.transitions), batch_size)
        return TransitionBatch(
            states=torch.from_numpy(np.stack([sample[0] for sample in samples])).to(
                device=device,
                dtype=torch.float32,
            ),
            actions=torch.tensor([sample[1] for sample in samples], device=device),
            rewards=torch.tensor(
                [sample[2] for sample in samples],
                device=device,
                dtype=torch.float32,
            ),
            next_states=torch.from_numpy(np.stack([sample[3] for sample in samples])).to(
                device=device,
                dtype=torch.float32,
            ),
            dones=torch.tensor(
                [sample[4] for sample in samples],
                device=device,
                dtype=torch.float32,
            ),
        )


class QNetwork(nn.Module):
    """Compact convolutional network that estimates one Q value per action."""

    def __init__(self, input_channels: int, action_count: int, grid_size: int) -> None:
        super().__init__()
        feature_dim = 16 * grid_size * grid_size
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_count),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class DQNAgent:
    """Online/target Q networks with epsilon-greedy action selection."""

    def __init__(
        self,
        input_channels: int,
        action_count: int,
        grid_size: int,
        *,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.input_channels = input_channels
        self.action_count = action_count
        self.grid_size = grid_size
        self.device = torch.device(device)
        self.online = QNetwork(input_channels, action_count, grid_size).to(self.device)
        self.target = QNetwork(input_channels, action_count, grid_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def select_action(
        self,
        observation: np.ndarray,
        *,
        epsilon: float,
        random_source: random.Random,
    ) -> int:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        if random_source.random() < epsilon:
            return random_source.randrange(self.action_count)
        state = (
            torch.from_numpy(observation)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=torch.float32,
            )
        )
        return int(self.online(state).argmax(dim=1).item())

    def update(self, batch: TransitionBatch, gamma: float = 0.99) -> dict[str, float]:
        q_values = self.online(batch.states)
        selected_q_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target(batch.next_states).max(dim=1).values
            targets = batch.rewards + gamma * (1.0 - batch.dones) * next_q_values

        loss = F.smooth_l1_loss(selected_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "q_mean": float(selected_q_values.detach().mean()),
            "target_mean": float(targets.detach().mean()),
            "gradient_norm": float(gradient_norm),
        }

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save_checkpoint(
        self,
        filename: str | Path,
        *,
        episode: int,
        total_steps: int,
        config: dict[str, Any],
    ) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "dqn",
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "episode": episode,
                "total_steps": total_steps,
                "config": config,
            },
            path,
        )

    def load_checkpoint(self, filename: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm") != "dqn":
            raise ValueError("checkpoint is not a DQN checkpoint")
        self.online.load_state_dict(checkpoint["online_state_dict"])
        self.target.load_state_dict(
            checkpoint.get("target_state_dict", checkpoint["online_state_dict"])
        )
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return {
            "episode": int(checkpoint.get("episode", -1)),
            "total_steps": int(checkpoint.get("total_steps", 0)),
            "config": dict(checkpoint.get("config", {})),
        }

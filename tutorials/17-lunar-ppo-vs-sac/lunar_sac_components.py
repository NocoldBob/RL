"""Soft Actor-Critic components for continuous-control lessons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBSERVATION_LOW = np.asarray([-1.2, -0.07], dtype=np.float32)
OBSERVATION_HIGH = np.asarray([0.6, 0.07], dtype=np.float32)
ACTION_EPSILON = 1e-6


def normalize_observation(observation: np.ndarray) -> np.ndarray:
    """Map Mountain Car position and velocity approximately to [-1, 1]."""
    observation = np.asarray(observation, dtype=np.float32)
    midpoint = (OBSERVATION_LOW + OBSERVATION_HIGH) / 2.0
    half_range = (OBSERVATION_HIGH - OBSERVATION_LOW) / 2.0
    return np.clip((observation - midpoint) / half_range, -1.5, 1.5).astype(np.float32)


class ReplayBuffer:
    """Fixed-size NumPy replay buffer with deterministic sampling."""

    def __init__(
        self,
        capacity: int,
        observation_size: int,
        action_size: int,
        *,
        seed: int,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.states = np.empty((capacity, observation_size), dtype=np.float32)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty((capacity, observation_size), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def append(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.states[self.position] = np.asarray(state, dtype=np.float32)
        self.actions[self.position] = np.asarray(action, dtype=np.float32)
        self.rewards[self.position] = float(reward)
        self.next_states[self.position] = np.asarray(next_state, dtype=np.float32)
        self.terminated[self.position] = float(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed the number of stored transitions")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return ReplayBatch(
            states=torch.as_tensor(self.states[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_states=torch.as_tensor(self.next_states[indices], device=device),
            terminated=torch.as_tensor(self.terminated[indices], device=device),
        )


@dataclass(slots=True)
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor


def build_mlp(input_size: int, output_size: int, hidden_size: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


class SquashedGaussianActor(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(states)
        return self.mean(features), self.log_std(features).clamp(-5.0, 2.0)

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(states)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_probabilities = distribution.log_prob(raw_actions).sum(dim=-1, keepdim=True)
        correction = torch.log(1.0 - actions.square() + ACTION_EPSILON).sum(dim=-1, keepdim=True)
        return actions, log_probabilities - correction

    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        mean, _ = self(states)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.model = build_mlp(observation_size + action_size, 1, hidden_size)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat((states, actions), dim=-1))


class SACAgent:
    def __init__(
        self,
        observation_size: int = 2,
        action_size: int = 1,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float | None = None,
        initial_alpha: float = 0.2,
        hidden_size: int = 64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.actor = SquashedGaussianActor(observation_size, action_size, hidden_size).to(
            self.device
        )
        self.q1 = QNetwork(observation_size, action_size, hidden_size).to(self.device)
        self.q2 = QNetwork(observation_size, action_size, hidden_size).to(self.device)
        self.target_q1 = QNetwork(observation_size, action_size, hidden_size).to(self.device)
        self.target_q2 = QNetwork(observation_size, action_size, hidden_size).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        for network in (self.target_q1, self.target_q2):
            network.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_optimizer = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()], lr=learning_rate
        )
        self.log_alpha = torch.tensor(
            np.log(initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = float(-action_size if target_entropy is None else target_entropy)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _state_tensor(self, observation: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(normalize_observation(observation), device=self.device).unsqueeze(0)

    @torch.no_grad()
    def predict(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        state = self._state_tensor(observation)
        if deterministic:
            action = self.actor.deterministic(state)
        else:
            action, _ = self.actor.sample(state)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        with torch.no_grad():
            next_actions, next_log_probabilities = self.actor.sample(batch.next_states)
            next_q = torch.minimum(
                self.target_q1(batch.next_states, next_actions),
                self.target_q2(batch.next_states, next_actions),
            )
            target = batch.rewards + self.gamma * (1.0 - batch.terminated) * (
                next_q - self.alpha.detach() * next_log_probabilities
            )

        q1_values = self.q1(batch.states, batch.actions)
        q2_values = self.q2(batch.states, batch.actions)
        q_loss = F.mse_loss(q1_values, target) + F.mse_loss(q2_values, target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        actions, log_probabilities = self.actor.sample(batch.states)
        actor_q = torch.minimum(self.q1(batch.states, actions), self.q2(batch.states, actions))
        actor_loss = (self.alpha.detach() * log_probabilities - actor_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probabilities.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        return {
            "q_loss": float(q_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "q1_mean": float(q1_values.detach().mean()),
            "q2_mean": float(q2_values.detach().mean()),
            "target_q_mean": float(target.detach().mean()),
            "q_disagreement": float((q1_values.detach() - q2_values.detach()).abs().mean()),
            "log_probability": float(log_probabilities.detach().mean()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                target_parameter.lerp_(source_parameter, self.tau)

    def save_checkpoint(self, path: str | Path, *, steps: int, config: dict[str, Any]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "sac",
                "actor_state_dict": self.actor.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "target_q1_state_dict": self.target_q1.state_dict(),
                "target_q2_state_dict": self.target_q2.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "q_optimizer_state_dict": self.q_optimizer.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "steps": steps,
                "config": config,
            },
            destination,
        )

    def load_checkpoint(self, path: str | Path, *, load_optimizers: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm") != "sac":
            raise ValueError("checkpoint is not a SAC checkpoint")
        for name in ("actor", "q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(checkpoint[f"{name}_state_dict"])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        return checkpoint

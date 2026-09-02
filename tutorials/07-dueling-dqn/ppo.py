"""Compact PPO agent and rollout utilities for the Snake lesson."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    next_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and value targets across episode boundaries."""

    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have equal length")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_non_terminal = 1.0 - float(dones[index])
        following_value = next_value if index == len(rewards) - 1 else float(values[index + 1])
        delta = (
            float(rewards[index])
            + gamma * following_value * next_non_terminal
            - float(values[index])
        )
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[index] = gae
    returns = advantages + values.astype(np.float32)
    return advantages, returns


class RolloutBuffer:
    """On-policy transitions collected before one PPO update."""

    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probabilities: list[float] = []
        self.values: list[float] = []

    def __len__(self) -> int:
        return len(self.states)

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_probability: float,
        value: float,
    ) -> None:
        self.states.append(state.copy())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probabilities.append(float(log_probability))
        self.values.append(float(value))

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probabilities.clear()
        self.values.clear()

    def build_batch(
        self,
        device: torch.device,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        if not self.states:
            raise ValueError("cannot build a batch from an empty rollout")
        advantages, returns = generalized_advantage_estimate(
            np.asarray(self.rewards, dtype=np.float32),
            np.asarray(self.values, dtype=np.float32),
            np.asarray(self.dones, dtype=np.float32),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantage_tensor = torch.from_numpy(advantages).to(device=device)
        if len(advantages) > 1:
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
                advantage_tensor.std(unbiased=False) + 1e-8
            )
        return RolloutBatch(
            states=torch.from_numpy(np.stack(self.states)).to(
                device=device,
                dtype=torch.float32,
            ),
            actions=torch.tensor(self.actions, device=device, dtype=torch.long),
            old_log_probabilities=torch.tensor(
                self.log_probabilities,
                device=device,
                dtype=torch.float32,
            ),
            returns=torch.from_numpy(returns).to(device=device, dtype=torch.float32),
            advantages=advantage_tensor,
        )


class PPOActorCritic(nn.Module):
    """Shared low-compute encoder with categorical policy and value heads."""

    def __init__(
        self,
        input_channels: int,
        action_count: int,
        grid_size: int,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        feature_dim = 16 * grid_size * grid_size
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64, action_count)
        self.critic = nn.Linear(64, 1)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observation)
        return self.actor(features), self.critic(features).squeeze(-1)

    @torch.no_grad()
    def predict(self, observation: torch.Tensor, *, deterministic: bool) -> int:
        logits, _ = self(observation)
        distribution = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return int(action.item())


class PPOAgent:
    """Categorical PPO using the same compact encoder as the Actor-Critic lesson."""

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
        self.device = torch.device(device)
        self.model = PPOActorCritic(
            input_channels,
            action_count,
            grid_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ).to(self.device)

    @torch.no_grad()
    def select_action(self, observation: np.ndarray) -> tuple[int, float, float]:
        state = (
            torch.from_numpy(observation)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=torch.float32,
            )
        )
        logits, value = self.model(state)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
        )

    @torch.no_grad()
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> int:
        state = (
            torch.from_numpy(observation)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=torch.float32,
            )
        )
        logits, _ = self.model(state)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        probabilities = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probabilities, 1, generator=generator).item())

    def update(
        self,
        batch: RolloutBatch,
        *,
        update_epochs: int,
        batch_size: int,
        clip_ratio: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
    ) -> dict[str, float]:
        sample_count = len(batch.actions)
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        updates = 0
        for _ in range(update_epochs):
            indices = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, batch_size):
                selected = indices[start : start + batch_size]
                logits, values = self.model(batch.states[selected])
                distribution = torch.distributions.Categorical(logits=logits)
                new_log_probabilities = distribution.log_prob(batch.actions[selected])
                log_ratio = new_log_probabilities - batch.old_log_probabilities[selected]
                ratio = log_ratio.exp()
                unclipped = ratio * batch.advantages[selected]
                clipped = (
                    ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * batch.advantages[selected]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * F.mse_loss(values, batch.returns[selected])
                entropy = distribution.entropy().mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                self.model.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                self.model.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
                metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approx_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                }
                for name, value in metrics.items():
                    totals[name] += float(value.detach())
                updates += 1
        return {name: value / updates for name, value in totals.items()}

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
                "algorithm": "ppo",
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.model.optimizer.state_dict(),
                "episode": episode,
                "total_steps": total_steps,
                "config": config,
            },
            path,
        )

    def load_checkpoint(self, filename: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm") != "ppo":
            raise ValueError("checkpoint is not a PPO checkpoint")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return {
            "episode": int(checkpoint.get("episode", -1)),
            "total_steps": int(checkpoint.get("total_steps", 0)),
            "config": dict(checkpoint.get("config", {})),
        }

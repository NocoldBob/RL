"""Continuous-action PPO components for MountainCarContinuous-v0."""

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
    """Map the two environment observations approximately to [-1, 1]."""
    observation = np.asarray(observation, dtype=np.float32)
    midpoint = (OBSERVATION_LOW + OBSERVATION_HIGH) / 2.0
    half_range = (OBSERVATION_HIGH - OBSERVATION_LOW) / 2.0
    return np.clip((observation - midpoint) / half_range, -1.5, 1.5).astype(np.float32)


def continuous_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_done: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE while bootstrapping truncations without crossing resets."""
    lengths = {len(rewards), len(values), len(next_values), len(terminated), len(episode_done)}
    if len(lengths) != 1:
        raise ValueError("all GAE arrays must have equal length")

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap = 1.0 - float(terminated[index])
        delta = (
            float(rewards[index])
            + gamma * bootstrap * float(next_values[index])
            - float(values[index])
        )
        continue_episode = 1.0 - float(episode_done[index])
        gae = delta + gamma * gae_lambda * continue_episode * gae
        advantages[index] = gae
    return advantages, advantages + values.astype(np.float32)


@dataclass(slots=True)
class ContinuousRolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class ContinuousRolloutBuffer:
    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.terminated: list[bool] = []
        self.episode_done: list[bool] = []
        self.log_probabilities: list[float] = []
        self.values: list[float] = []
        self.next_values: list[float] = []

    def __len__(self) -> int:
        return len(self.states)

    def append(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        episode_done: bool,
        log_probability: float,
        value: float,
        next_value: float,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.episode_done.append(bool(episode_done))
        self.log_probabilities.append(float(log_probability))
        self.values.append(float(value))
        self.next_values.append(float(next_value))

    def clear(self) -> None:
        self.__init__()

    def build_batch(
        self,
        device: torch.device,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> ContinuousRolloutBatch:
        if not self.states:
            raise ValueError("cannot build a batch from an empty rollout")
        advantages, returns = continuous_gae(
            np.asarray(self.rewards, dtype=np.float32),
            np.asarray(self.values, dtype=np.float32),
            np.asarray(self.next_values, dtype=np.float32),
            np.asarray(self.terminated, dtype=np.float32),
            np.asarray(self.episode_done, dtype=np.float32),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantage_tensor = torch.as_tensor(advantages, device=device)
        if len(advantages) > 1:
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
                advantage_tensor.std(unbiased=False) + 1e-8
            )
        return ContinuousRolloutBatch(
            states=torch.as_tensor(np.stack(self.states), device=device),
            actions=torch.as_tensor(np.stack(self.actions), device=device),
            old_log_probabilities=torch.as_tensor(self.log_probabilities, device=device),
            returns=torch.as_tensor(returns, device=device),
            advantages=advantage_tensor,
        )


class GaussianActorCritic(nn.Module):
    """Small actor-critic with a tanh-squashed Gaussian policy."""

    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(states)
        mean = self.actor_mean(features)
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        value = self.critic(features).squeeze(-1)
        return mean, log_std, value

    @staticmethod
    def squash_log_probability(
        distribution: torch.distributions.Normal,
        raw_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        base_log_probability = distribution.log_prob(raw_actions).sum(dim=-1)
        correction = torch.log(1.0 - actions.square() + ACTION_EPSILON).sum(dim=-1)
        return base_log_probability - correction

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, values = self(states)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_probabilities = self.squash_log_probability(distribution, raw_actions, actions)
        return actions, log_probabilities, values

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, values = self(states)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        safe_actions = actions.clamp(-1.0 + ACTION_EPSILON, 1.0 - ACTION_EPSILON)
        raw_actions = torch.atanh(safe_actions)
        log_probabilities = self.squash_log_probability(distribution, raw_actions, safe_actions)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probabilities, entropy, values


class ContinuousPPOAgent:
    def __init__(
        self,
        observation_size: int = 2,
        action_size: int = 1,
        *,
        learning_rate: float = 3e-4,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = GaussianActorCritic(observation_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _state_tensor(self, observation: np.ndarray) -> torch.Tensor:
        normalized = normalize_observation(observation)
        return torch.as_tensor(normalized, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def select_action(self, observation: np.ndarray) -> tuple[np.ndarray, float, float]:
        actions, log_probabilities, values = self.model.sample(self._state_tensor(observation))
        return (
            actions.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_probabilities.item()),
            float(values.item()),
        )

    @torch.no_grad()
    def value(self, observation: np.ndarray) -> float:
        _, _, value = self.model(self._state_tensor(observation))
        return float(value.item())

    @torch.no_grad()
    def predict(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        state = self._state_tensor(observation)
        mean, _, _ = self.model(state)
        if deterministic:
            action = torch.tanh(mean)
        else:
            action, _, _ = self.model.sample(state)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(
        self,
        batch: ContinuousRolloutBatch,
        *,
        update_epochs: int,
        batch_size: int,
        clip_ratio: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
    ) -> dict[str, float]:
        totals = {
            name: 0.0
            for name in (
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
            )
        }
        updates = 0
        sample_count = len(batch.actions)
        for _ in range(update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, batch_size):
                indices = permutation[start : start + batch_size]
                new_log_probabilities, entropy, values = self.model.evaluate_actions(
                    batch.states[indices], batch.actions[indices]
                )
                log_ratio = new_log_probabilities - batch.old_log_probabilities[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * batch.advantages[indices]
                clipped = (
                    ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * batch.advantages[indices]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * F.mse_loss(values, batch.returns[indices])
                entropy_mean = entropy.mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
                values_to_record = (
                    loss,
                    policy_loss,
                    value_loss,
                    entropy_mean,
                    approximate_kl,
                    clip_fraction,
                )
                for name, metric in zip(totals, values_to_record, strict=True):
                    totals[name] += float(metric.detach())
                updates += 1
        return {name: total / updates for name, total in totals.items()}

    def save_checkpoint(self, path: str | Path, *, steps: int, config: dict[str, Any]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "continuous_ppo",
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps": steps,
                "config": config,
            },
            destination,
        )

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm") != "continuous_ppo":
            raise ValueError("checkpoint is not a continuous PPO checkpoint")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

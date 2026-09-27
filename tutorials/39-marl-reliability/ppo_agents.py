"""Shared IPPO, MAPPO, and differentiable communication policies."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from value_agents import (
    ACTION_COUNT,
    GLOBAL_STATE_SIZE,
    LOCAL_OBSERVATION_SIZE,
    padded_observation,
    padded_state,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(slots=True)
class PPOConfig:
    mode: str = "ippo"
    agents: int = 3
    total_steps: int = 30_000
    rollout_steps: int = 500
    epochs: int = 5
    batch_size: int = 256
    gamma: float = 0.95
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    message_dropout: float = 0.0
    eval_interval: int = 5_000
    eval_episodes: int = 30
    seed: int = 42
    output_dir: Path = Path("runs/marl-ppo/default")

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


class SharedMultiAgentActorCritic(nn.Module):
    def __init__(self, agents: int, mode: str) -> None:
        super().__init__()
        if mode not in {"ippo", "mappo", "communication"}:
            raise ValueError(f"unsupported PPO mode: {mode}")
        self.agents = agents
        self.mode = mode
        self.message_size = 8 if mode == "communication" else 0
        if self.message_size:
            self.message_encoder = nn.Sequential(
                nn.Linear(LOCAL_OBSERVATION_SIZE, 32),
                nn.Tanh(),
                nn.Linear(32, self.message_size),
                nn.Tanh(),
            )
        actor_input = LOCAL_OBSERVATION_SIZE + 4 + self.message_size
        self.actor = nn.Sequential(
            nn.Linear(actor_input, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, ACTION_COUNT),
        )
        if mode == "ippo":
            self.critic = nn.Sequential(
                nn.Linear(LOCAL_OBSERVATION_SIZE + 4, 128), nn.Tanh(), nn.Linear(128, 1)
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(GLOBAL_STATE_SIZE, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, agents),
            )

    @staticmethod
    def identities(batch: int, agents: int, device: torch.device) -> torch.Tensor:
        return torch.eye(4, device=device)[:agents].unsqueeze(0).expand(batch, -1, -1)

    def actor_input(
        self,
        observations: torch.Tensor,
        *,
        message_dropout: float = 0.0,
    ) -> torch.Tensor:
        batch, agents, _ = observations.shape
        identity = self.identities(batch, agents, observations.device)
        parts = [observations, identity]
        if self.message_size:
            messages = self.message_encoder(observations)
            received = (messages.sum(dim=1, keepdim=True) - messages) / max(1, agents - 1)
            if message_dropout > 0:
                mask = (
                    torch.rand(batch, agents, 1, device=observations.device) >= message_dropout
                ).float()
                received = received * mask
            parts.append(received)
        return torch.cat(parts, dim=-1)

    def distributions(
        self,
        observations: torch.Tensor,
        *,
        message_dropout: float = 0.0,
    ) -> Categorical:
        return Categorical(
            logits=self.actor(self.actor_input(observations, message_dropout=message_dropout))
        )

    def values(self, observations: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        if self.mode == "ippo":
            batch, agents, _ = observations.shape
            identity = self.identities(batch, agents, observations.device)
            return self.critic(torch.cat([observations, identity], dim=-1)).squeeze(-1)
        return self.critic(states)


class MultiAgentRollout:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probabilities: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.next_values: list[np.ndarray] = []
        self.done: list[bool] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.__init__()

    def build(
        self, device: torch.device, gamma: float, gae_lambda: float
    ) -> dict[str, torch.Tensor]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        next_values = np.asarray(self.next_values, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = np.zeros(rewards.shape[1], dtype=np.float32)
        for index in range(len(rewards) - 1, -1, -1):
            continuation = 1.0 - float(self.done[index])
            delta = rewards[index] + gamma * continuation * next_values[index] - values[index]
            gae = delta + gamma * gae_lambda * continuation * gae
            advantages[index] = gae
        returns = advantages + values
        advantage_tensor = torch.as_tensor(advantages, device=device)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std(unbiased=False) + 1e-8
        )
        return {
            "observations": torch.as_tensor(np.stack(self.observations), device=device),
            "states": torch.as_tensor(np.stack(self.states), device=device),
            "actions": torch.as_tensor(np.stack(self.actions), device=device),
            "old_log_probabilities": torch.as_tensor(
                np.stack(self.log_probabilities), device=device
            ),
            "returns": torch.as_tensor(returns, device=device),
            "advantages": advantage_tensor,
        }


class MultiAgentPPO:
    def __init__(self, config: PPOConfig, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = SharedMultiAgentActorCritic(config.agents, config.mode).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def prepare(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack(
            [
                padded_observation(observations[f"agent_{index}"])
                for index in range(self.config.agents)
            ]
        )

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        state: np.ndarray,
        *,
        deterministic: bool = False,
        message_dropout: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observation_tensor = torch.as_tensor(observations, device=self.device).unsqueeze(0)
        state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
        dropout = self.config.message_dropout if message_dropout is None else message_dropout
        distribution = self.model.distributions(observation_tensor, message_dropout=dropout)
        actions = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        values = self.model.values(observation_tensor, state_tensor)
        return (
            actions[0].cpu().numpy(),
            distribution.log_prob(actions)[0].cpu().numpy(),
            values[0].cpu().numpy(),
        )

    def policy(self, observations: dict[str, np.ndarray], environment) -> dict[str, int]:
        prepared = self.prepare(observations)
        actions, _, _ = self.act(
            prepared, padded_state(environment.state()), deterministic=True, message_dropout=0.0
        )
        return {f"agent_{index}": int(action) for index, action in enumerate(actions)}

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        totals = {name: 0.0 for name in ("loss", "policy_loss", "value_loss", "entropy", "kl")}
        updates = 0
        time_steps = len(batch["actions"])
        for _ in range(self.config.epochs):
            permutation = torch.randperm(time_steps, device=self.device)
            for start in range(0, time_steps, self.config.batch_size):
                indices = permutation[start : start + self.config.batch_size]
                observations = batch["observations"][indices]
                distribution = self.model.distributions(
                    observations, message_dropout=self.config.message_dropout
                )
                log_probabilities = distribution.log_prob(batch["actions"][indices])
                entropy = distribution.entropy()
                values = self.model.values(observations, batch["states"][indices])
                log_ratio = log_probabilities - batch["old_log_probabilities"][indices]
                ratio = log_ratio.exp()
                advantages = batch["advantages"][indices]
                clipped = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
                value_loss = (values - batch["returns"][indices]).square().mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy.mean()
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                values_to_add = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy.mean(),
                    "kl": ((ratio - 1) - log_ratio).mean(),
                }
                for name, value in values_to_add.items():
                    totals[name] += float(value.detach().item())
                updates += 1
        return {name: value / updates for name, value in totals.items()}

    def save(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self.model.state_dict(), "config": self.config.payload(), "step": step}, path
        )

    def load(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        return int(payload["step"])

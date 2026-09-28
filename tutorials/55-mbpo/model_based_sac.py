"""Small generic SAC and replay utilities for Dyna/MBPO comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamics_model import DynamicsDataset, DynamicsModel


@dataclass(slots=True)
class Batch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor


class Replay:
    def __init__(self, capacity: int = 200_000, seed: int = 0) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.observations = np.empty((capacity, 3), dtype=np.float32)
        self.actions = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty((capacity, 3), dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def append(self, observation, action, reward, next_observation) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend_dataset(self, dataset: DynamicsDataset) -> None:
        for index in range(len(dataset)):
            self.append(
                dataset.observations[index],
                dataset.actions[index] / 2.0,
                dataset.rewards[index] * 0.1,
                dataset.next_observations[index],
            )

    def sample(self, size: int) -> Batch:
        indices = self.rng.integers(0, self.size, size=size)
        return Batch(
            observations=torch.as_tensor(self.observations[indices]),
            actions=torch.as_tensor(self.actions[indices]),
            rewards=torch.as_tensor(self.rewards[indices, None]),
            next_observations=torch.as_tensor(self.next_observations[indices]),
        )


def merge(first: Batch, second: Batch) -> Batch:
    return Batch(
        **{
            field: torch.cat([getattr(first, field), getattr(second, field)], dim=0)
            for field in Batch.__annotations__
        }
    )


def mlp(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, outputs)
    )


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = mlp(3, 2)

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.body(observations).chunk(2, dim=1)
        log_std = log_std.clamp(-5, 1)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_probability = distribution.log_prob(raw) - torch.log(1 - action.square() + 1e-6)
        return action, log_probability

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = self.body(observations).chunk(2, dim=1)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = mlp(4, 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([observations, actions], dim=1))


class SimpleSAC:
    def __init__(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.actor = Actor()
        self.q1, self.q2 = Critic(), Critic()
        self.target_q1, self.target_q2 = Critic(), Critic()
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q_optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=3e-4)
        self.alpha = 0.1

    def update(self, batch: Batch) -> dict[str, float]:
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(batch.next_observations)
            next_q = torch.minimum(
                self.target_q1(batch.next_observations, next_actions),
                self.target_q2(batch.next_observations, next_actions),
            )
            targets = batch.rewards + 0.99 * (next_q - self.alpha * next_log_prob)
        q1 = self.q1(batch.observations, batch.actions)
        q2 = self.q2(batch.observations, batch.actions)
        q_loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        actions, log_prob = self.actor.sample(batch.observations)
        actor_loss = (
            self.alpha * log_prob
            - torch.minimum(
                self.q1(batch.observations, actions), self.q2(batch.observations, actions)
            )
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        with torch.no_grad():
            for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                for parameter, target_parameter in zip(
                    source.parameters(), target.parameters(), strict=True
                ):
                    target_parameter.lerp_(parameter, 0.01)
        return {"q_loss": float(q_loss.detach()), "actor_loss": float(actor_loss.detach())}

    @torch.no_grad()
    def normalized_action(
        self, observations: torch.Tensor, stochastic: bool = False
    ) -> torch.Tensor:
        return (
            self.actor.sample(observations)[0]
            if stochastic
            else self.actor.deterministic(observations)
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        return (2.0 * self.normalized_action(tensor)[0].numpy()).astype(np.float32)

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load(cls, path: Path) -> tuple[SimpleSAC, dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls()
        agent.actor.load_state_dict(payload["actor"])
        return agent, payload["metadata"]


@torch.no_grad()
def add_imagined_rollouts(
    replay: Replay,
    real: DynamicsDataset,
    model: DynamicsModel,
    agent: SimpleSAC,
    *,
    starts: int,
    horizon: int,
    rng: np.random.Generator,
) -> None:
    indices = rng.integers(0, len(real), size=starts)
    observations = torch.as_tensor(real.observations[indices])
    for _ in range(horizon):
        normalized_actions = agent.normalized_action(observations, stochastic=True)
        physical_actions = normalized_actions * 2.0
        next_observations, rewards = model.predict(observations, physical_actions)
        for index in range(starts):
            replay.append(
                observations[index].numpy(),
                normalized_actions[index].numpy(),
                float(rewards[index]) * 0.1,
                next_observations[index].numpy(),
            )
        observations = next_observations


def train_with_imagination(
    real: DynamicsDataset,
    model: DynamicsModel,
    *,
    imagined_fraction: float,
    rollout_horizon: int,
    updates: int = 6_000,
    seed: int = 0,
) -> tuple[SimpleSAC, dict[str, Any]]:
    real_replay = Replay(seed=seed)
    real_replay.extend_dataset(real)
    imagined_replay = Replay(seed=seed + 1)
    agent = SimpleSAC(seed)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(1, updates + 1):
        if imagined_fraction > 0 and (step == 1 or step % 250 == 0):
            add_imagined_rollouts(
                imagined_replay, real, model, agent, starts=256, horizon=rollout_horizon, rng=rng
            )
        imagined_size = int(256 * imagined_fraction) if len(imagined_replay) else 0
        real_batch = real_replay.sample(256 - imagined_size)
        batch = (
            merge(real_batch, imagined_replay.sample(imagined_size))
            if imagined_size
            else real_batch
        )
        metrics = agent.update(batch)
        if step % 500 == 0:
            history.append({"step": step, "imagined_transitions": len(imagined_replay), **metrics})
    return agent, {
        "imagined_fraction": imagined_fraction,
        "rollout_horizon": rollout_horizon,
        "updates": updates,
        "history": history,
    }

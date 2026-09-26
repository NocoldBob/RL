"""CNN PPO components for continuous visual driving."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from cnn_encoder import NatureEncoder

ACTION_EPSILON = 1e-6


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    done: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = (
            rewards[index] + gamma * (1.0 - terminated[index]) * next_values[index] - values[index]
        )
        gae = delta + gamma * gae_lambda * (1.0 - done[index]) * gae
        advantages[index] = gae
    return advantages, advantages + values


def augment_observations(observations: torch.Tensor) -> torch.Tensor:
    """Cheap batch-wise brightness and integer translation augmentation."""
    gain = torch.empty((len(observations), 1, 1, 1), device=observations.device).uniform_(0.8, 1.2)
    augmented = (observations * gain).clamp(0.0, 1.0)
    shift_y = int(torch.randint(-2, 3, ()).item())
    shift_x = int(torch.randint(-2, 3, ()).item())
    return torch.roll(augmented, shifts=(shift_y, shift_x), dims=(-2, -1))


class VisualGaussianActorCritic(nn.Module):
    def __init__(self, channels: int, image_size: int = 64) -> None:
        super().__init__()
        self.encoder = NatureEncoder(channels, image_size)
        self.actor_mean = nn.Linear(256, 3)
        self.critic = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.asarray([-0.5, -0.5, -1.0]))
        with torch.no_grad():
            self.actor_mean.bias.copy_(torch.asarray([0.0, 0.0, -2.0]))

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        mean = self.actor_mean(features)
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        values = self.critic(features).squeeze(-1)
        return mean, log_std, values

    @staticmethod
    def _log_probability(
        distribution: torch.distributions.Normal,
        raw_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return distribution.log_prob(raw_actions).sum(-1) - torch.log(
            1.0 - actions.square() + ACTION_EPSILON
        ).sum(-1)

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, values = self(observations)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions)
        return actions, self._log_probability(distribution, raw_actions, actions), values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, values = self(observations)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        safe_actions = actions.clamp(-1 + ACTION_EPSILON, 1 - ACTION_EPSILON)
        raw_actions = torch.atanh(safe_actions)
        log_probabilities = self._log_probability(distribution, raw_actions, safe_actions)
        return log_probabilities, distribution.entropy().sum(-1), values


@dataclass(slots=True)
class PPOConfig:
    frame_stack: int = 4
    image_size: int = 64
    total_steps: int = 100_000
    rollout_steps: int = 512
    epochs: int = 4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1
    reward_mode: str = "raw"
    augmentation: bool = False
    eval_interval: int = 10_000
    eval_episodes: int = 5
    seed: int = 42
    output_dir: Path = Path("runs/carracing-ppo/default")

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


class VisualRollout:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.next_values: list[float] = []
        self.log_probabilities: list[float] = []
        self.terminated: list[bool] = []
        self.done: list[bool] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.__init__()

    def build(
        self,
        device: torch.device,
        gamma: float,
        gae_lambda: float,
    ) -> dict[str, torch.Tensor]:
        advantages, returns = generalized_advantage_estimate(
            np.asarray(self.rewards, dtype=np.float32),
            np.asarray(self.values, dtype=np.float32),
            np.asarray(self.next_values, dtype=np.float32),
            np.asarray(self.terminated, dtype=np.float32),
            np.asarray(self.done, dtype=np.float32),
            gamma,
            gae_lambda,
        )
        advantage_tensor = torch.as_tensor(advantages, device=device)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std(unbiased=False) + 1e-8
        )
        return {
            "observations": torch.as_tensor(np.stack(self.observations), device=device),
            "actions": torch.as_tensor(np.stack(self.actions), device=device),
            "old_log_probabilities": torch.as_tensor(self.log_probabilities, device=device),
            "returns": torch.as_tensor(returns, device=device),
            "advantages": advantage_tensor,
        }


class VisualPPOAgent:
    def __init__(
        self,
        channels: int,
        *,
        image_size: int = 64,
        learning_rate: float = 2.5e-4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = VisualGaussianActorCritic(channels, image_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _tensor(self, observation: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(observation, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> tuple[np.ndarray, float, float]:
        actions, log_probabilities, values = self.model.sample(self._tensor(observation))
        return (
            actions[0].cpu().numpy().astype(np.float32),
            float(log_probabilities.item()),
            float(values.item()),
        )

    @torch.no_grad()
    def value(self, observation: np.ndarray) -> float:
        return float(self.model(self._tensor(observation))[2].item())

    @torch.no_grad()
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        mean, _, _ = self.model(self._tensor(observation))
        if deterministic:
            actions = torch.tanh(mean)
        else:
            actions, _, _ = self.model.sample(self._tensor(observation))
        return actions[0].cpu().numpy().astype(np.float32)

    def update(
        self,
        batch: dict[str, torch.Tensor],
        config: PPOConfig,
    ) -> dict[str, float]:
        totals = {name: 0.0 for name in ("loss", "policy_loss", "value_loss", "entropy", "kl")}
        updates = 0
        count = len(batch["actions"])
        for _ in range(config.epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, config.batch_size):
                indices = permutation[start : start + config.batch_size]
                observations = batch["observations"][indices]
                if config.augmentation:
                    observations = augment_observations(observations)
                log_probabilities, entropy, values = self.model.evaluate_actions(
                    observations, batch["actions"][indices]
                )
                log_ratio = log_probabilities - batch["old_log_probabilities"][indices]
                ratio = log_ratio.exp()
                advantages = batch["advantages"][indices]
                policy_loss = -torch.minimum(
                    ratio * advantages,
                    ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantages,
                ).mean()
                value_loss = (values - batch["returns"][indices]).square().mean()
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy.mean()
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()
                metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy.mean(),
                    "kl": ((ratio - 1) - log_ratio).mean(),
                }
                for name, value in metrics.items():
                    totals[name] += float(value.detach().item())
                updates += 1
        return {name: value / updates for name, value in totals.items()}

    def save(self, path: Path, *, steps: int, config: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "steps": steps, "config": config}, path)

    def load(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        return int(payload.get("steps", 0))

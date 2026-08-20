"""Small convolutional Actor-Critic network used by the Snake lesson."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvActorCritic(nn.Module):
    """A compact shared encoder with categorical actor and state-value critic."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        grid_size: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        entropy_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.entropy_coef = entropy_coef
        feature_dim = 16 * grid_size * grid_size
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.actor = nn.Linear(feature_dim, output_dim)
        self.critic = nn.Linear(feature_dim, 1)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(observation)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def predict(self, observation: torch.Tensor, deterministic: bool = False) -> int:
        logits, _ = self(observation)
        distribution = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return int(action.item())

    def update(
        self,
        state: torch.Tensor,
        action: int | torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        gamma: float = 0.99,
    ) -> dict[str, float]:
        """Apply one correct one-step Actor-Critic update."""

        logits, value = self(state)
        with torch.no_grad():
            _, next_value = self(next_state)
            target = torch.as_tensor(reward, dtype=value.dtype, device=value.device)
            if not done:
                target = target + gamma * next_value

        action_tensor = torch.as_tensor(action, dtype=torch.long, device=value.device).reshape(-1)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probability = distribution.log_prob(action_tensor)
        advantage = target - value
        actor_loss = -(log_probability * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.square().mean()
        entropy = distribution.entropy().mean()
        loss = actor_loss + critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "entropy": float(entropy.detach()),
        }

    def imitation_update(
        self,
        state: torch.Tensor,
        teacher_action: int | torch.Tensor,
    ) -> float:
        """Learn an explicitly supplied teacher action with behavior cloning."""

        logits, _ = self(state)
        target = torch.as_tensor(
            teacher_action,
            dtype=torch.long,
            device=logits.device,
        ).reshape(-1)
        loss = F.cross_entropy(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.detach())

    def value_update(
        self,
        state: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        gamma: float = 0.99,
    ) -> float:
        """Train the critic on a teacher transition without a policy-gradient update."""

        _, value = self(state)
        with torch.no_grad():
            _, next_value = self(next_state)
            target = torch.as_tensor(reward, dtype=value.dtype, device=value.device)
            if not done:
                target = target + gamma * next_value
        loss = 0.5 * (target - value).square().mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.detach())

    def save_checkpoint(
        self,
        filename: str | Path,
        *,
        episode: int,
        config: dict[str, Any],
    ) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "episode": episode,
                "config": config,
            },
            path,
        )

    def load_checkpoint(self, filename: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return {
            "episode": int(checkpoint.get("episode", -1)),
            "config": dict(checkpoint.get("config", {})),
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

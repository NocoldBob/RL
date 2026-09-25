"""PPO and SAC agents adapted to LunarLanderContinuous-v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from lunar_ppo_components import ContinuousPPOAgent, GaussianActorCritic
from lunar_sac_components import SACAgent

OBSERVATION_SCALE = np.asarray([1.0, 1.5, 2.0, 2.0, np.pi, 5.0, 1.0, 1.0], dtype=np.float32)


def normalize_lunar_observation(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (8,):
        raise ValueError("LunarLander observation must have shape (8,)")
    return np.clip(observation / OBSERVATION_SCALE, -5.0, 5.0).astype(np.float32)


class LunarGaussianActorCritic(GaussianActorCritic):
    """Separate actor and critic encoders to reduce gradient interference."""

    def __init__(self, *, initial_log_std: float = -0.5) -> None:
        nn.Module.__init__(self)
        self.actor_encoder = nn.Sequential(
            nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.critic_encoder = nn.Sequential(
            nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.actor_mean = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((2,), initial_log_std))
        for module in (*self.actor_encoder, *self.critic_encoder):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_features = self.actor_encoder(states)
        critic_features = self.critic_encoder(states)
        mean = self.actor_mean(actor_features)
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        value = self.critic(critic_features).squeeze(-1)
        return mean, log_std, value


class LunarPPOAgent(ContinuousPPOAgent):
    def __init__(
        self,
        *,
        learning_rate: float = 3e-4,
        initial_log_std: float = -0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(8, 2, learning_rate=learning_rate, device=device)
        self.model = LunarGaussianActorCritic(initial_log_std=initial_log_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)

    def _state_tensor(self, observation: np.ndarray) -> torch.Tensor:
        normalized = normalize_lunar_observation(observation)
        return torch.as_tensor(normalized, device=self.device).unsqueeze(0)

    def save_checkpoint(self, path: str | Path, *, steps: int, config: dict[str, Any]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "lunar_ppo",
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps": steps,
                "config": config,
            },
            destination,
        )

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm") != "lunar_ppo":
            raise ValueError("checkpoint is not a LunarLander PPO checkpoint")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint


class LunarSACAgent(SACAgent):
    def __init__(
        self,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        initial_alpha: float = 0.2,
        target_entropy: float | None = None,
        hidden_size: int = 128,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            8,
            2,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            initial_alpha=initial_alpha,
            target_entropy=target_entropy,
            hidden_size=hidden_size,
            device=device,
        )

    def _state_tensor(self, observation: np.ndarray) -> torch.Tensor:
        normalized = normalize_lunar_observation(observation)
        return torch.as_tensor(normalized, device=self.device).unsqueeze(0)

    def save_checkpoint(self, path: str | Path, *, steps: int, config: dict[str, Any]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "lunar_sac",
                "actor_state_dict": self.actor.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "target_q1_state_dict": self.target_q1.state_dict(),
                "target_q2_state_dict": self.target_q2.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "steps": steps,
                "config": config,
            },
            destination,
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm") != "lunar_sac":
            raise ValueError("checkpoint is not a LunarLander SAC checkpoint")
        for name in ("actor", "q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(checkpoint[f"{name}_state_dict"])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"])
        return checkpoint

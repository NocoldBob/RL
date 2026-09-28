"""Behavior cloning, offline DQN, discrete CQL, and discrete IQL."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from acrobot_env import ACTION_COUNT, OBSERVATION_SIZE
from online_dqn import QNetwork


class PolicyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_COUNT),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class ValueNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


class OfflineAgent:
    def __init__(
        self,
        mode: str,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        cql_alpha: float = 1.0,
        expectile: float = 0.7,
        advantage_beta: float = 3.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if mode not in {"bc", "dqn", "cql", "iql"}:
            raise ValueError(f"unsupported offline mode: {mode}")
        self.mode = mode
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.expectile = expectile
        self.advantage_beta = advantage_beta
        self.device = torch.device(device)
        self.policy = PolicyNetwork().to(self.device) if mode in {"bc", "iql"} else None
        self.q = QNetwork().to(self.device) if mode != "bc" else None
        self.target_q = copy.deepcopy(self.q) if self.q is not None else None
        self.value = ValueNetwork().to(self.device) if mode == "iql" else None
        parameters: list[nn.Parameter] = []
        for module in (self.policy, self.q, self.value):
            if module is not None:
                parameters.extend(module.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> int:
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        logits = self.policy(tensor) if self.policy is not None else self.q(tensor)
        return int(logits.argmax(dim=1).item())

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        observations = batch["observations"]
        actions = batch["actions"]
        if self.mode == "bc":
            assert self.policy is not None
            logits = self.policy(observations)
            loss = F.cross_entropy(logits, actions)
            metrics = {
                "loss": loss,
                "accuracy": (logits.argmax(dim=1) == actions).float().mean(),
            }
        elif self.mode in {"dqn", "cql"}:
            assert self.q is not None and self.target_q is not None
            q_values = self.q(observations)
            chosen = q_values.gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                next_actions = self.q(batch["next_observations"]).argmax(dim=1, keepdim=True)
                next_values = (
                    self.target_q(batch["next_observations"]).gather(1, next_actions).squeeze(1)
                )
                targets = batch["rewards"] + self.gamma * (1 - batch["terminated"]) * next_values
            bellman = F.smooth_l1_loss(chosen, targets)
            conservative = torch.logsumexp(q_values, dim=1).mean() - chosen.mean()
            loss = bellman + (self.cql_alpha * conservative if self.mode == "cql" else 0.0)
            data_mask = F.one_hot(actions, ACTION_COUNT).bool()
            unseen = q_values.masked_fill(data_mask, -torch.inf).max(dim=1).values
            metrics = {
                "loss": loss,
                "bellman_loss": bellman,
                "conservative_loss": conservative,
                "data_q": chosen.mean(),
                "unseen_q": unseen.mean(),
                "q_gap": (unseen - chosen).mean(),
            }
        else:
            assert self.q is not None and self.target_q is not None
            assert self.value is not None and self.policy is not None
            with torch.no_grad():
                target_chosen = self.target_q(observations).gather(1, actions[:, None]).squeeze(1)
            value_prediction = self.value(observations)
            difference = target_chosen - value_prediction
            weights = torch.where(difference > 0, self.expectile, 1 - self.expectile)
            value_loss = (weights * difference.square()).mean()
            with torch.no_grad():
                next_value = self.value(batch["next_observations"])
                targets = batch["rewards"] + self.gamma * (1 - batch["terminated"]) * next_value
            chosen = self.q(observations).gather(1, actions[:, None]).squeeze(1)
            q_loss = F.smooth_l1_loss(chosen, targets)
            with torch.no_grad():
                advantages = target_chosen - value_prediction
                policy_weights = (self.advantage_beta * advantages).exp().clamp(max=100.0)
            logits = self.policy(observations)
            policy_loss = (
                policy_weights * F.cross_entropy(logits, actions, reduction="none")
            ).mean()
            loss = q_loss + value_loss + policy_loss
            metrics = {
                "loss": loss,
                "q_loss": q_loss,
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "advantage": advantages.mean(),
                "accuracy": (logits.argmax(dim=1) == actions).float().mean(),
            }
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [parameter for group in self.optimizer.param_groups for parameter in group["params"]],
            10.0,
        )
        self.optimizer.step()
        return {name: float(value.detach().item()) for name, value in metrics.items()}

    def sync_target(self) -> None:
        if self.q is not None and self.target_q is not None:
            self.target_q.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def validation_metrics(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        actions = batch["actions"]
        if self.policy is not None:
            logits = self.policy(batch["observations"])
            return {
                "validation_loss": float(F.cross_entropy(logits, actions).item()),
                "validation_accuracy": float((logits.argmax(1) == actions).float().mean().item()),
            }
        assert self.q is not None and self.target_q is not None
        q_values = self.q(batch["observations"])
        chosen = q_values.gather(1, actions[:, None]).squeeze(1)
        next_values = self.target_q(batch["next_observations"]).max(dim=1).values
        targets = batch["rewards"] + self.gamma * (1 - batch["terminated"]) * next_values
        mask = F.one_hot(actions, ACTION_COUNT).bool()
        unseen = q_values.masked_fill(mask, -torch.inf).max(dim=1).values
        return {
            "validation_loss": float(F.smooth_l1_loss(chosen, targets).item()),
            "data_q": float(chosen.mean().item()),
            "unseen_q": float(unseen.mean().item()),
            "q_gap": float((unseen - chosen).mean().item()),
        }

    def save(self, path: Path, step: int, config: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": self.mode,
                "policy": None if self.policy is None else self.policy.state_dict(),
                "q": None if self.q is None else self.q.state_dict(),
                "value": None if self.value is None else self.value.state_dict(),
                "step": step,
                "config": config,
            },
            path,
        )

    def load(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if self.policy is not None and payload["policy"] is not None:
            self.policy.load_state_dict(payload["policy"])
        if self.q is not None and payload["q"] is not None:
            self.q.load_state_dict(payload["q"])
            assert self.target_q is not None
            self.target_q.load_state_dict(payload["q"])
        if self.value is not None and payload["value"] is not None:
            self.value.load_state_dict(payload["value"])
        return int(payload["step"])

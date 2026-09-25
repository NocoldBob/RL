"""Feed-forward and GRU PPO agents for partial-observation DoorKey."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from doorkey_dqn import ENCODED_CHANNELS, encode_grid_tensor, image_observation
from doorkey_env import make_env


class GridActorCritic(nn.Module):
    def __init__(self, *, recurrent: bool) -> None:
        super().__init__()
        self.recurrent = recurrent
        self.encoder = nn.Sequential(
            nn.Conv2d(ENCODED_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.memory = nn.GRUCell(64, 64) if recurrent else None
        self.actor = nn.Linear(64, 7)
        self.critic = nn.Linear(64, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(
        self, observations: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(encode_grid_tensor(observations))
        next_hidden = self.memory(features, hidden) if self.memory is not None else features
        return self.actor(next_hidden), self.critic(next_hidden).squeeze(1), next_hidden


class DoorKeyPPOAgent:
    def __init__(
        self,
        *,
        recurrent: bool,
        learning_rate: float = 2.5e-4,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.recurrent = recurrent
        self.model = GridActorCritic(recurrent=recurrent).to(self.device)
        self.reference_model: GridActorCritic | None = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, 64, device=self.device)

    def set_reference_policy(self) -> None:
        self.reference_model = copy.deepcopy(self.model).eval()
        self.reference_model.requires_grad_(False)

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        hidden: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[int, float, float, torch.Tensor]:
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        logits, value, next_hidden = self.model(tensor, hidden)
        distribution = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=1) if deterministic else distribution.sample()
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
            next_hidden,
        )

    @torch.no_grad()
    def value(self, observation: np.ndarray, hidden: torch.Tensor) -> float:
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        _, value, _ = self.model(tensor, hidden)
        return float(value.item())

    def update(
        self,
        batch: RolloutBatch,
        *,
        epochs: int,
        batch_size: int,
        clip_ratio: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        target_kl: float | None,
        reference_kl_coef: float = 0.0,
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
        epochs_completed = 0
        early_stopped = False
        for _ in range(epochs):
            if self.recurrent:
                sequence_length = 32
                starts = list(range(0, len(batch.actions), sequence_length))
                random.shuffle(starts)
                selections = [
                    torch.arange(
                        start,
                        min(start + sequence_length, len(batch.actions)),
                        device=self.device,
                    )
                    for start in starts
                ]
            else:
                indices = torch.randperm(len(batch.actions), device=self.device)
                selections = [
                    indices[start : start + batch_size]
                    for start in range(0, len(indices), batch_size)
                ]
            epoch_kls = []
            for selected in selections:
                if self.recurrent:
                    hidden = batch.hidden[selected[0]].unsqueeze(0)
                    logits_list = []
                    values_list = []
                    for index in selected:
                        logits_step, value_step, hidden = self.model(
                            batch.observations[index].unsqueeze(0), hidden
                        )
                        logits_list.append(logits_step)
                        values_list.append(value_step)
                        if batch.done[index]:
                            hidden = self.initial_hidden()
                    logits = torch.cat(logits_list)
                    values = torch.cat(values_list)
                else:
                    logits, values, _ = self.model(
                        batch.observations[selected], batch.hidden[selected]
                    )
                distribution = torch.distributions.Categorical(logits=logits)
                log_probabilities = distribution.log_prob(batch.actions[selected])
                log_ratio = log_probabilities - batch.old_log_probabilities[selected]
                ratio = log_ratio.exp()
                unclipped = ratio * batch.advantages[selected]
                clipped = (
                    ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * batch.advantages[selected]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (values - batch.returns[selected]).square().mean()
                entropy = distribution.entropy().mean()
                reference_kl = torch.tensor(0.0, device=self.device)
                if self.reference_model is not None and reference_kl_coef > 0:
                    with torch.no_grad():
                        reference_logits, _, _ = self.reference_model(
                            batch.observations[selected], batch.hidden[selected]
                        )
                    reference_distribution = torch.distributions.Categorical(
                        logits=reference_logits
                    )
                    reference_kl = torch.distributions.kl_divergence(
                        reference_distribution, distribution
                    ).mean()
                loss = (
                    policy_loss
                    + value_coef * value_loss
                    - entropy_coef * entropy
                    + reference_kl_coef * reference_kl
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
                for name, metric in zip(
                    totals,
                    (loss, policy_loss, value_loss, entropy, approximate_kl, clip_fraction),
                    strict=True,
                ):
                    totals[name] += float(metric.detach())
                epoch_kls.append(float(approximate_kl))
                updates += 1
            epochs_completed += 1
            if target_kl is not None and np.mean(epoch_kls) > target_kl:
                early_stopped = True
                break
        result = {name: value / updates for name, value in totals.items()}
        result.update(
            {
                "epochs_completed": float(epochs_completed),
                "early_stopped": float(early_stopped),
            }
        )
        return result

    def save(self, path: Path, *, steps: int, config: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "doorkey_gru_ppo" if self.recurrent else "doorkey_ppo",
                "model_state_dict": self.model.state_dict(),
                "steps": steps,
                "config": config,
            },
            path,
        )

    def load(self, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint


@dataclass(slots=True)
class RolloutBatch:
    observations: torch.Tensor
    hidden: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    done: torch.Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.hidden: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probabilities: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.next_values: list[float] = []
        self.terminated: list[bool] = []
        self.done: list[bool] = []

    def __len__(self) -> int:
        return len(self.actions)

    def clear(self) -> None:
        self.__init__()

    def build(self, device: torch.device, *, gamma: float, gae_lambda: float) -> RolloutBatch:
        advantages = np.zeros(len(self), dtype=np.float32)
        gae = 0.0
        for index in range(len(self) - 1, -1, -1):
            delta = (
                self.rewards[index]
                + gamma * (1.0 - self.terminated[index]) * self.next_values[index]
                - self.values[index]
            )
            gae = delta + gamma * gae_lambda * (1.0 - self.done[index]) * gae
            advantages[index] = gae
        returns = advantages + np.asarray(self.values, dtype=np.float32)
        advantage_tensor = torch.as_tensor(advantages, device=device)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std(unbiased=False) + 1e-8
        )
        return RolloutBatch(
            observations=torch.as_tensor(np.stack(self.observations), device=device),
            hidden=torch.as_tensor(np.stack(self.hidden), device=device),
            actions=torch.as_tensor(self.actions, device=device),
            old_log_probabilities=torch.as_tensor(self.log_probabilities, device=device),
            returns=torch.as_tensor(returns, device=device),
            advantages=advantage_tensor,
            done=torch.as_tensor(self.done, device=device),
        )


@dataclass(slots=True)
class PPOConfig:
    recurrent: bool = False
    size: int = 5
    curriculum: bool = False
    count_bonus: float = 0.0
    reference_kl_coef: float = 0.0
    total_steps: int = 100_000
    rollout_steps: int = 1024
    epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    eval_interval: int = 10_000
    eval_episodes: int = 50
    seed: int = 42
    initial_checkpoint: Path | None = None
    output_dir: Path = Path("runs/doorkey-ppo/feedforward-seed-42")

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        result["initial_checkpoint"] = (
            str(self.initial_checkpoint) if self.initial_checkpoint is not None else None
        )
        return result


def evaluate_ppo(
    agent: DoorKeyPPOAgent, seeds: list[int], *, size: int = 5
) -> dict[str, float | int]:
    env = make_env(size, observation_mode="compact_partial")
    successes = returns = steps_total = 0.0
    try:
        for seed in seeds:
            raw_observation, _ = env.reset(seed=seed)
            observation = image_observation(raw_observation)
            hidden = agent.initial_hidden()
            terminated = truncated = False
            episode_return = 0.0
            steps = 0
            while not (terminated or truncated):
                action, _, _, hidden = agent.act(observation, hidden, deterministic=True)
                raw_observation, reward, terminated, truncated, _ = env.step(action)
                observation = image_observation(raw_observation)
                episode_return += float(reward)
                steps += 1
            successes += float(terminated and episode_return > 0)
            returns += episode_return
            steps_total += steps
    finally:
        env.close()
    count = len(seeds)
    return {
        "episodes": count,
        "success_rate": successes / count,
        "average_return": returns / count,
        "average_steps": steps_total / count,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

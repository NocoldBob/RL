"""Fitted Q Evaluation and trajectory-wise importance sampling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dataset import OfflineDataset
from offline_agents import OfflineAgent
from online_dqn import QNetwork


def target_probabilities(
    agent: OfflineAgent, observations: np.ndarray, epsilon: float = 0.05
) -> np.ndarray:
    device = agent.device
    with torch.no_grad():
        tensor = torch.as_tensor(observations, device=device)
        scores = agent.policy(tensor) if agent.policy is not None else agent.q(tensor)
        greedy = scores.argmax(dim=1).cpu().numpy()
    probabilities = np.full((len(observations), 3), epsilon / 3, dtype=np.float64)
    probabilities[np.arange(len(observations)), greedy] += 1 - epsilon
    return probabilities


def importance_sampling(
    dataset: OfflineDataset,
    agent: OfflineAgent,
    *,
    gamma: float = 0.99,
    target_epsilon: float = 0.05,
) -> dict[str, float]:
    target = target_probabilities(agent, dataset.observations, target_epsilon)
    chosen_target = target[np.arange(len(dataset)), dataset.actions]
    ratios = chosen_target / np.clip(dataset.behavior_probability, 1e-8, None)
    episodes = np.unique(dataset.episode_id)
    log_weights = []
    returns = []
    for episode in episodes:
        indices = np.flatnonzero(dataset.episode_id == episode)
        log_weights.append(float(np.log(np.clip(ratios[indices], 1e-12, None)).sum()))
        discounts = np.power(gamma, np.arange(len(indices)))
        returns.append(float(np.sum(discounts * dataset.rewards[indices])))
    log_weights_array = np.asarray(log_weights)
    stabilized = np.exp(np.clip(log_weights_array - log_weights_array.max(), -50, 0))
    returns_array = np.asarray(returns)
    wis = float(np.sum(stabilized * returns_array) / np.sum(stabilized))
    raw_weights = np.exp(np.clip(log_weights_array, -50, 20))
    ordinary = float(np.mean(raw_weights * returns_array))
    effective_size = float(stabilized.sum() ** 2 / np.square(stabilized).sum())
    return {
        "ordinary_is": ordinary,
        "weighted_is": wis,
        "effective_sample_size": effective_size,
        "maximum_log_weight": float(log_weights_array.max()),
        "minimum_log_weight": float(log_weights_array.min()),
    }


class FittedQEvaluator:
    def __init__(self, gamma: float = 0.99, learning_rate: float = 3e-4) -> None:
        self.gamma = gamma
        self.q = QNetwork()
        self.target = QNetwork()
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)

    def train(
        self,
        dataset: OfflineDataset,
        agent: OfflineAgent,
        *,
        gradient_steps: int = 10_000,
        batch_size: int = 256,
    ) -> list[dict[str, float]]:
        history = []
        for step in range(1, gradient_steps + 1):
            batch = dataset.sample(batch_size, torch.device("cpu"))
            predicted = (
                self.q(batch["observations"]).gather(1, batch["actions"][:, None]).squeeze(1)
            )
            with torch.no_grad():
                next_numpy = batch["next_observations"].cpu().numpy()
                probabilities = torch.as_tensor(
                    target_probabilities(agent, next_numpy), dtype=torch.float32
                )
                next_value = (probabilities * self.target(batch["next_observations"])).sum(dim=1)
                targets = batch["rewards"] + self.gamma * (1 - batch["terminated"]) * next_value
            loss = F.smooth_l1_loss(predicted, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if step % 250 == 0:
                self.target.load_state_dict(self.q.state_dict())
            if step % 500 == 0:
                history.append({"step": float(step), "loss": float(loss.item())})
        return history

    @torch.no_grad()
    def estimate(self, dataset: OfflineDataset, agent: OfflineAgent) -> float:
        first_indices = np.asarray(
            [
                np.flatnonzero(dataset.episode_id == episode)[0]
                for episode in np.unique(dataset.episode_id)
            ]
        )
        observations = dataset.observations[first_indices]
        probabilities = torch.as_tensor(
            target_probabilities(agent, observations), dtype=torch.float32
        )
        return float(
            (probabilities * self.q(torch.as_tensor(observations))).sum(dim=1).mean().item()
        )

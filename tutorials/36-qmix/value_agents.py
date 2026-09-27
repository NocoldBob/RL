"""IQL, parameter-sharing DQN, VDN, and QMIX for discrete cooperative control."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_AGENTS = 4
LOCAL_OBSERVATION_SIZE = 24
GLOBAL_STATE_SIZE = 96
ACTION_COUNT = 5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def padded_observation(observation: np.ndarray, agent_id: int | None = None) -> np.ndarray:
    vector = np.zeros(LOCAL_OBSERVATION_SIZE, dtype=np.float32)
    source = np.asarray(observation, dtype=np.float32)
    vector[: min(len(vector), len(source))] = source[: len(vector)]
    if agent_id is None:
        return vector
    identity = np.zeros(MAX_AGENTS, dtype=np.float32)
    identity[agent_id] = 1.0
    return np.concatenate([vector, identity])


def padded_state(state: np.ndarray) -> np.ndarray:
    vector = np.zeros(GLOBAL_STATE_SIZE, dtype=np.float32)
    source = np.asarray(state, dtype=np.float32)
    vector[: min(len(vector), len(source))] = source[: len(vector)]
    return vector


class AgentQNetwork(nn.Module):
    def __init__(self, observation_size: int = LOCAL_OBSERVATION_SIZE) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_COUNT),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class QMixer(nn.Module):
    """Monotonic QMIX mixer with non-negative hypernetwork weights."""

    def __init__(
        self, agents: int, state_size: int = GLOBAL_STATE_SIZE, hidden_size: int = 32
    ) -> None:
        super().__init__()
        self.agents = agents
        self.hidden_size = hidden_size
        self.hyper_w1 = nn.Linear(state_size, agents * hidden_size)
        self.hyper_b1 = nn.Linear(state_size, hidden_size)
        self.hyper_w2 = nn.Linear(state_size, hidden_size)
        self.value = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, agent_values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch = len(agent_values)
        w1 = self.hyper_w1(states).abs().view(batch, self.agents, self.hidden_size)
        b1 = self.hyper_b1(states).view(batch, 1, self.hidden_size)
        hidden = F.elu(torch.bmm(agent_values.unsqueeze(1), w1) + b1)
        w2 = self.hyper_w2(states).abs().view(batch, self.hidden_size, 1)
        return torch.bmm(hidden, w2).squeeze(-1).squeeze(-1) + self.value(states).squeeze(-1)


class JointReplayBuffer:
    def __init__(self, capacity: int, agents: int, observation_size: int) -> None:
        self.capacity = capacity
        self.agents = agents
        self.observations = np.empty((capacity, agents, observation_size), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.states = np.empty((capacity, GLOBAL_STATE_SIZE), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty((capacity, agents), dtype=np.int64)
        self.rewards = np.empty((capacity, agents), dtype=np.float32)
        self.done = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def append(
        self,
        observations: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.observations[index] = observations
        self.states[index] = state
        self.actions[index] = actions
        self.rewards[index] = rewards
        self.next_observations[index] = next_observations
        self.next_states[index] = next_state
        self.done[index] = done
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[indices], device=device),
            "states": torch.as_tensor(self.states[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_observations": torch.as_tensor(self.next_observations[indices], device=device),
            "next_states": torch.as_tensor(self.next_states[indices], device=device),
            "done": torch.as_tensor(self.done[indices], device=device),
        }


@dataclass(slots=True)
class ValueConfig:
    mode: str = "iql"
    agents: int = 3
    agent_id: bool = False
    total_steps: int = 20_000
    replay_capacity: int = 30_000
    warmup_steps: int = 1_000
    batch_size: int = 64
    gamma: float = 0.95
    learning_rate: float = 3e-4
    target_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 15_000
    eval_interval: int = 5_000
    eval_episodes: int = 30
    seed: int = 42
    output_dir: Path = Path("runs/marl-value/default")

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


class MultiAgentValueLearner:
    def __init__(self, config: ValueConfig, device: str | torch.device = "cpu") -> None:
        if config.mode not in {"iql", "shared", "vdn", "qmix"}:
            raise ValueError(f"unsupported mode: {config.mode}")
        self.config = config
        self.device = torch.device(device)
        observation_size = LOCAL_OBSERVATION_SIZE + (MAX_AGENTS if config.agent_id else 0)
        network_count = config.agents if config.mode == "iql" else 1
        self.networks = nn.ModuleList(
            [AgentQNetwork(observation_size) for _ in range(network_count)]
        ).to(self.device)
        self.targets = copy.deepcopy(self.networks).to(self.device)
        self.mixer = QMixer(config.agents).to(self.device) if config.mode == "qmix" else None
        self.target_mixer = (
            copy.deepcopy(self.mixer).to(self.device) if self.mixer is not None else None
        )
        parameters = list(self.networks.parameters())
        if self.mixer is not None:
            parameters += list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)

    @property
    def observation_size(self) -> int:
        return LOCAL_OBSERVATION_SIZE + (MAX_AGENTS if self.config.agent_id else 0)

    def prepare(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack(
            [
                padded_observation(
                    observations[f"agent_{index}"], index if self.config.agent_id else None
                )
                for index in range(self.config.agents)
            ]
        )

    @torch.no_grad()
    def act_array(self, observations: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        actions = []
        for index in range(self.config.agents):
            if np.random.random() < epsilon:
                actions.append(np.random.randint(ACTION_COUNT))
                continue
            network = self.networks[index] if self.config.mode == "iql" else self.networks[0]
            tensor = torch.as_tensor(observations[index], device=self.device).unsqueeze(0)
            actions.append(int(network(tensor).argmax(dim=1).item()))
        return np.asarray(actions, dtype=np.int64)

    def policy(self, observations: dict[str, np.ndarray], _environment=None) -> dict[str, int]:
        actions = self.act_array(self.prepare(observations))
        return {f"agent_{index}": int(action) for index, action in enumerate(actions)}

    def _agent_q(self, observations: torch.Tensor, target: bool = False) -> torch.Tensor:
        networks = self.targets if target else self.networks
        if self.config.mode == "iql":
            return torch.stack(
                [networks[i](observations[:, i]) for i in range(self.config.agents)], dim=1
            )
        batch, agents, features = observations.shape
        return networks[0](observations.reshape(batch * agents, features)).view(
            batch, agents, ACTION_COUNT
        )

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        q_all = self._agent_q(batch["observations"])
        chosen = q_all.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_online = self._agent_q(batch["next_observations"])
            next_actions = next_online.argmax(-1, keepdim=True)
            next_target = self._agent_q(batch["next_observations"], target=True)
            next_chosen = next_target.gather(-1, next_actions).squeeze(-1)
        if self.config.mode in {"iql", "shared"}:
            targets = (
                batch["rewards"] + self.config.gamma * (1 - batch["done"][:, None]) * next_chosen
            )
            loss = F.smooth_l1_loss(chosen, targets)
            predicted_mean = chosen.mean()
        else:
            team_reward = batch["rewards"].mean(dim=1)
            if self.config.mode == "vdn":
                predicted = chosen.sum(dim=1)
                next_total = next_chosen.sum(dim=1)
            else:
                assert self.mixer is not None and self.target_mixer is not None
                predicted = self.mixer(chosen, batch["states"])
                with torch.no_grad():
                    next_total = self.target_mixer(next_chosen, batch["next_states"])
            target = team_reward + self.config.gamma * (1 - batch["done"]) * next_total
            loss = F.smooth_l1_loss(predicted, target)
            predicted_mean = predicted.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.networks.parameters(), 10.0)
        self.optimizer.step()
        return {"loss": float(loss.item()), "q_mean": float(predicted_mean.item())}

    def sync_targets(self) -> None:
        self.targets.load_state_dict(self.networks.state_dict())
        if self.mixer is not None and self.target_mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "networks": self.networks.state_dict(),
                "mixer": None if self.mixer is None else self.mixer.state_dict(),
                "config": self.config.payload(),
                "step": step,
            },
            path,
        )

    def load(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.networks.load_state_dict(payload["networks"])
        self.targets.load_state_dict(payload["networks"])
        if self.mixer is not None and payload["mixer"] is not None:
            self.mixer.load_state_dict(payload["mixer"])
            assert self.target_mixer is not None
            self.target_mixer.load_state_dict(payload["mixer"])
        return int(payload["step"])

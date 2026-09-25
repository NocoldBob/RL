"""Compact full-map DQN for MiniGrid DoorKey."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from doorkey_env import make_env
from minigrid.core.world_object import Door, Key

OBJECT_CHANNELS = 11
COLOR_CHANNELS = 6
STATE_CHANNELS = 4
ENCODED_CHANNELS = OBJECT_CHANNELS + COLOR_CHANNELS + STATE_CHANNELS


def encode_grid_tensor(observations: torch.Tensor) -> torch.Tensor:
    """Convert compact HxWx3 MiniGrid codes into channel-first one-hot tensors."""
    observations = observations.long()
    objects = F.one_hot(observations[..., 0], OBJECT_CHANNELS)
    colors = F.one_hot(observations[..., 1], COLOR_CHANNELS)
    states = F.one_hot(observations[..., 2].clamp(max=STATE_CHANNELS - 1), STATE_CHANNELS)
    return torch.cat((objects, colors, states), dim=-1).permute(0, 3, 1, 2).float()


class GridQNetwork(nn.Module):
    def __init__(self, height: int, width: int, action_count: int = 7) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(ENCODED_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, action_count)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(encode_grid_tensor(observations)))


class GridReplayBuffer:
    def __init__(self, capacity: int, shape: tuple[int, int, int], *, seed: int) -> None:
        self.capacity = capacity
        self.states = np.empty((capacity, *shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *shape), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.terminated[self.position] = float(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.terminated[indices], device=device),
        )


class DoorKeyDQNAgent:
    def __init__(
        self,
        height: int,
        width: int,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.online = GridQNetwork(height, width).to(self.device)
        self.target = GridQNetwork(height, width).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

    @torch.no_grad()
    def predict(self, observation: np.ndarray) -> int:
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        return int(self.online(tensor).argmax(dim=1).item())

    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        states, actions, rewards, next_states, terminated = batch
        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * (1.0 - terminated) * next_q
        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "q_mean": float(q_values.detach().mean()),
            "target_mean": float(targets.detach().mean()),
            "gradient_norm": float(gradient_norm),
        }

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: Path, *, steps: int, config: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "doorkey_dqn",
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "steps": steps,
                "config": config,
            },
            path,
        )

    def load(self, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(checkpoint["online_state_dict"])
        self.target.load_state_dict(checkpoint["target_state_dict"])
        return checkpoint


@dataclass(slots=True)
class DQNConfig:
    size: int = 5
    total_steps: int = 100_000
    replay_capacity: int = 50_000
    learning_starts: int = 2_000
    batch_size: int = 64
    update_frequency: int = 4
    target_sync_interval: int = 1_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 60_000
    reward_mode: str = "sparse"
    observation_mode: str = "compact_full"
    eval_interval: int = 10_000
    eval_episodes: int = 50
    seed: int = 42
    output_dir: Path = Path("runs/doorkey-dqn/sparse-seed-42")

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result


def door_is_open(env: Any) -> bool:
    base = env.unwrapped
    for x in range(base.width):
        for y in range(base.height):
            cell = base.grid.get(x, y)
            if isinstance(cell, Door):
                return bool(cell.is_open)
    raise RuntimeError("door not found")


def shaped_reward(
    raw_reward: float,
    *,
    picked_before: bool,
    picked_after: bool,
    door_before: bool,
    door_after: bool,
) -> float:
    reward = raw_reward - 0.001
    if picked_after and not picked_before:
        reward += 0.2
    if door_after and not door_before:
        reward += 0.2
    return reward


def image_observation(observation: Any) -> np.ndarray:
    return observation["image"] if isinstance(observation, dict) else observation


def evaluate_dqn(
    agent: DoorKeyDQNAgent,
    size: int,
    seeds: list[int],
    *,
    observation_mode: str = "compact_full",
) -> dict[str, float | int]:
    env = make_env(size, observation_mode=observation_mode)
    successes = key_pickups = door_opens = returns = steps_total = 0.0
    try:
        for seed in seeds:
            raw_observation, _ = env.reset(seed=seed)
            observation = image_observation(raw_observation)
            terminated = truncated = False
            episode_return = 0.0
            steps = 0
            picked_key = opened_door = False
            while not (terminated or truncated):
                action = agent.predict(observation)
                raw_observation, reward, terminated, truncated, _ = env.step(action)
                observation = image_observation(raw_observation)
                episode_return += float(reward)
                steps += 1
                picked_key = picked_key or isinstance(env.unwrapped.carrying, Key)
                opened_door = opened_door or door_is_open(env)
            successes += float(terminated and episode_return > 0)
            key_pickups += float(picked_key)
            door_opens += float(opened_door)
            returns += episode_return
            steps_total += steps
    finally:
        env.close()
    count = len(seeds)
    return {
        "episodes": count,
        "success_rate": successes / count,
        "key_pickup_rate": key_pickups / count,
        "door_open_rate": door_opens / count,
        "average_return": returns / count,
        "average_steps": steps_total / count,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

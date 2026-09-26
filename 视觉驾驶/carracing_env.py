"""Shared CarRacing environment, preprocessing, baselines, and evaluation."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

DISCRETE_ACTIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-0.6, 0.8, 0.0],
        [0.6, 0.8, 0.0],
        [0.0, 0.0, 0.8],
        [-0.8, 0.0, 0.6],
        [0.8, 0.0, 0.6],
    ],
    dtype=np.float32,
)


def preprocess_frame(
    frame: np.ndarray,
    *,
    size: int = 64,
    grayscale: bool = True,
) -> np.ndarray:
    """Remove the HUD, resize with deterministic nearest sampling, and normalize."""
    if frame.shape != (96, 96, 3):
        raise ValueError(f"expected a 96x96 RGB frame, got {frame.shape}")
    cropped = frame[:84]
    y_index = np.linspace(0, cropped.shape[0] - 1, size).astype(np.int64)
    x_index = np.linspace(0, cropped.shape[1] - 1, size).astype(np.int64)
    resized = cropped[y_index][:, x_index].astype(np.float32) / 255.0
    if grayscale:
        gray = np.dot(resized[..., :3], np.asarray([0.299, 0.587, 0.114], dtype=np.float32))
        return gray[None].astype(np.float32)
    return np.moveaxis(resized, -1, 0).astype(np.float32)


def corrupt_frame(
    frame: np.ndarray,
    kind: str,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a deterministic-test-friendly visual corruption before preprocessing."""
    if kind == "none" or severity <= 0:
        return frame
    image = frame.astype(np.float32)
    strength = float(np.clip(severity, 0.0, 1.0))
    if kind == "brightness":
        image *= 1.0 - 0.65 * strength
    elif kind == "noise":
        image += rng.normal(0.0, 55.0 * strength, image.shape)
    elif kind == "color":
        gains = np.asarray([1 + 0.5 * strength, 1 - 0.45 * strength, 1 + 0.15 * strength])
        image *= gains
    elif kind == "occlusion":
        width = max(1, int(40 * strength))
        start = (image.shape[1] - width) // 2
        image[18:64, start : start + width] = 0
    elif kind == "blur":
        shifted = [np.roll(image, offset, axis=axis) for axis in (0, 1) for offset in (-1, 1)]
        image = (image + strength * sum(shifted)) / (1 + 4 * strength)
    else:
        raise ValueError(f"unknown corruption: {kind}")
    return np.clip(image, 0, 255).astype(np.uint8)


def continuous_action_from_policy(action: np.ndarray) -> np.ndarray:
    """Map a policy vector from [-1, 1] to CarRacing's mixed action bounds."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (3,):
        raise ValueError("continuous policy action must have shape (3,)")
    return np.asarray(
        [
            np.clip(action[0], -1.0, 1.0),
            np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0),
            np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


@dataclass(slots=True)
class DrivingConfig:
    frame_stack: int = 4
    frame_skip: int = 4
    image_size: int = 64
    grayscale: bool = True
    action_mode: str = "continuous"
    reward_mode: str = "raw"
    corruption: str = "none"
    corruption_severity: float = 0.0
    max_decisions: int = 250
    seed: int = 42


class VisualCarRacing:
    """A small explicit wrapper used identically by DQN, PPO, and benchmarks."""

    def __init__(self, config: DrivingConfig) -> None:
        if config.frame_stack < 1 or config.frame_skip < 1:
            raise ValueError("frame_stack and frame_skip must be positive")
        if config.action_mode not in {"continuous", "discrete"}:
            raise ValueError("action_mode must be continuous or discrete")
        if config.reward_mode not in {"raw", "shaped"}:
            raise ValueError("reward_mode must be raw or shaped")
        self.config = config
        self.env = gym.make("CarRacing-v3", continuous=True)
        self.frames: deque[np.ndarray] = deque(maxlen=config.frame_stack)
        self.rng = np.random.default_rng(config.seed)
        self.decisions = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.raw_return = 0.0
        self.offroad_frames = 0
        self.action_jitter = 0.0

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        channels = 1 if self.config.grayscale else 3
        return (channels * self.config.frame_stack, self.config.image_size, self.config.image_size)

    @property
    def action_count(self) -> int:
        return len(DISCRETE_ACTIONS)

    def _process(self, raw: np.ndarray) -> np.ndarray:
        corrupted = corrupt_frame(
            raw,
            self.config.corruption,
            self.config.corruption_severity,
            self.rng,
        )
        return preprocess_frame(
            corrupted,
            size=self.config.image_size,
            grayscale=self.config.grayscale,
        )

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)

    @staticmethod
    def _offroad_fraction(raw: np.ndarray) -> float:
        # The lower central crop is where the car and nearby road appear.  A
        # whole-frame grass ratio mostly measures scenery, not whether the car left the road.
        road = raw[56:78, 34:62]
        green = (road[..., 1] > road[..., 0] * 1.15) & (road[..., 1] > road[..., 2] * 1.15)
        return float(green.mean())

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        actual_seed = self.config.seed if seed is None else seed
        raw, info = self.env.reset(seed=actual_seed)
        processed = self._process(raw)
        self.frames.clear()
        for _ in range(self.config.frame_stack):
            self.frames.append(processed.copy())
        self.decisions = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.raw_return = 0.0
        self.offroad_frames = 0
        self.action_jitter = 0.0
        return self._stack(), info

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        env_action = (
            DISCRETE_ACTIONS[int(action)].copy()
            if self.config.action_mode == "discrete"
            else continuous_action_from_policy(np.asarray(action))
        )
        reward_sum = 0.0
        raw_reward_sum = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}
        raw: np.ndarray | None = None
        offroad = 0.0
        for _ in range(self.config.frame_skip):
            raw, reward, terminated, truncated, info = self.env.step(env_action)
            raw_reward_sum += float(reward)
            offroad = self._offroad_fraction(raw)
            shaped = float(reward)
            if self.config.reward_mode == "shaped":
                shaped -= 0.04 * abs(float(env_action[0]))
                shaped -= 0.08 * float(env_action[2])
                shaped -= 0.15 * max(0.0, offroad - 0.55)
            reward_sum += shaped
            if terminated or truncated:
                break
        assert raw is not None
        self.frames.append(self._process(raw))
        self.decisions += 1
        self.raw_return += raw_reward_sum
        self.offroad_frames += int(offroad > 0.55)
        self.action_jitter += float(np.abs(env_action - self.last_action).mean())
        self.last_action = env_action
        if self.decisions >= self.config.max_decisions and not (terminated or truncated):
            truncated = True
        info = dict(info)
        info.update(
            {
                "raw_reward": raw_reward_sum,
                "raw_return": self.raw_return,
                "offroad_fraction": offroad,
                "offroad_rate": self.offroad_frames / self.decisions,
                "action_jitter": self.action_jitter / self.decisions,
                "env_action": env_action.copy(),
            }
        )
        return self._stack(), reward_sum, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


def privileged_track_action(environment: VisualCarRacing) -> np.ndarray:
    """Steer toward a nearby track segment using simulator state, for a ceiling baseline."""
    base = environment.env.unwrapped
    car = base.car
    if car is None or not base.track:
        return np.asarray([0.0, -0.2, -1.0], dtype=np.float32)
    position = np.asarray(car.hull.position, dtype=np.float32)
    points = np.asarray([[row[2], row[3]] for row in base.track], dtype=np.float32)
    nearest = int(np.argmin(np.square(points - position).sum(axis=1)))
    target = points[(nearest + 5) % len(points)]
    desired = math.atan2(float(target[1] - position[1]), float(target[0] - position[0]))
    # Box2D's hull angle is measured along the car's local x-axis, while the
    # car points along local y.  The pi/2 correction avoids an immediate full-lock turn.
    heading = float(car.hull.angle) + math.pi / 2
    error = (desired - heading + math.pi) % (2 * math.pi) - math.pi
    steer = float(np.clip(-1.8 * error, -1.0, 1.0))
    speed = float(np.linalg.norm(car.hull.linearVelocity))
    gas = 0.8 if speed < 28 else 0.2
    brake = 0.5 if abs(error) > 0.65 and speed > 18 else 0.0
    return np.asarray([steer, 2 * gas - 1, 2 * brake - 1], dtype=np.float32)


def run_episode(
    environment: VisualCarRacing,
    policy: Callable[[np.ndarray], int | np.ndarray],
    *,
    seed: int,
) -> dict[str, float | int | bool]:
    observation, _ = environment.reset(seed=seed)
    total_reward = 0.0
    info: dict[str, Any] = {}
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = policy(observation)
        observation, reward, terminated, truncated, info = environment.step(action)
        total_reward += float(reward)
        steps += 1
    raw_return = float(info.get("raw_return", total_reward))
    return {
        "seed": seed,
        "return": raw_return,
        "learning_return": total_reward,
        "steps": steps,
        "success": raw_return >= 200.0,
        "offroad_rate": float(info.get("offroad_rate", 0.0)),
        "action_jitter": float(info.get("action_jitter", 0.0)),
    }


def summarize_episodes(rows: list[dict[str, float | int | bool]]) -> dict[str, float]:
    return {
        "episodes": float(len(rows)),
        "average_return": float(np.mean([float(row["return"]) for row in rows])),
        "return_std": float(np.std([float(row["return"]) for row in rows])),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "average_steps": float(np.mean([int(row["steps"]) for row in rows])),
        "offroad_rate": float(np.mean([float(row["offroad_rate"]) for row in rows])),
        "action_jitter": float(np.mean([float(row["action_jitter"]) for row in rows])),
    }

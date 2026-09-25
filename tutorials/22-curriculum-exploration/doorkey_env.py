"""MiniGrid DoorKey helpers shared by lessons 18-24."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401 -- importing registers MiniGrid environments
import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgPartialObsWrapper

DIRECTION_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))
POLICIES = ("random", "forward", "expert")


def env_id(size: int) -> str:
    if size not in {5, 6, 8, 16}:
        raise ValueError("DoorKey size must be one of 5, 6, 8, or 16")
    return f"MiniGrid-DoorKey-{size}x{size}-v0"


def make_env(
    size: int,
    *,
    observation_mode: str = "compact_partial",
    render_mode: str | None = None,
) -> gym.Env:
    env = gym.make(env_id(size), render_mode=render_mode)
    if observation_mode == "compact_partial":
        return env
    if observation_mode == "compact_full":
        return ImgObsWrapper(FullyObsWrapper(env))
    if observation_mode == "rgb_partial":
        return ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=8))
    raise ValueError(f"unknown observation mode: {observation_mode}")


@dataclass(frozen=True, slots=True)
class PlannerState:
    x: int
    y: int
    direction: int
    has_key: bool
    door_open: bool


@dataclass(slots=True)
class EpisodeResult:
    policy: str
    size: int
    seed: int
    success: bool
    episode_return: float
    steps: int
    picked_key: bool
    opened_door: bool
    timeout: bool

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def locate_objects(env: gym.Env) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    base = env.unwrapped
    key_position = door_position = goal_position = None
    for x in range(base.width):
        for y in range(base.height):
            cell = base.grid.get(x, y)
            if isinstance(cell, Key):
                key_position = (x, y)
            elif isinstance(cell, Door):
                door_position = (x, y)
            elif isinstance(cell, Goal):
                goal_position = (x, y)
    if key_position is None or door_position is None or goal_position is None:
        raise RuntimeError("DoorKey grid is missing a key, door, or goal")
    return key_position, door_position, goal_position


def expert_plan(env: gym.Env) -> list[int]:
    """Find a shortest valid action sequence using the privileged full grid."""
    base = env.unwrapped
    key_position, door_position, goal_position = locate_objects(env)
    door = base.grid.get(*door_position)
    start = PlannerState(
        int(base.agent_pos[0]),
        int(base.agent_pos[1]),
        int(base.agent_dir),
        isinstance(base.carrying, Key),
        bool(door.is_open),
    )
    queue = deque([(start, [])])
    visited = {start}
    actions = (Actions.left, Actions.right, Actions.forward, Actions.pickup, Actions.toggle)

    def transition(state: PlannerState, action: Actions) -> PlannerState | None:
        if action == Actions.left:
            return PlannerState(
                state.x, state.y, (state.direction - 1) % 4, state.has_key, state.door_open
            )
        if action == Actions.right:
            return PlannerState(
                state.x, state.y, (state.direction + 1) % 4, state.has_key, state.door_open
            )
        dx, dy = DIRECTION_VECTORS[state.direction]
        front = (state.x + dx, state.y + dy)
        if action == Actions.pickup:
            if not state.has_key and front == key_position:
                return PlannerState(state.x, state.y, state.direction, True, state.door_open)
            return None
        if action == Actions.toggle:
            if state.has_key and front == door_position and not state.door_open:
                return PlannerState(state.x, state.y, state.direction, True, True)
            return None
        if action != Actions.forward:
            return None
        cell = base.grid.get(*front)
        if isinstance(cell, Wall):
            return None
        if front == key_position and not state.has_key:
            return None
        if front == door_position and not state.door_open:
            return None
        return PlannerState(front[0], front[1], state.direction, state.has_key, state.door_open)

    while queue:
        state, plan = queue.popleft()
        if (state.x, state.y) == goal_position:
            return plan
        for action in actions:
            next_state = transition(state, action)
            if next_state is not None and next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, [*plan, int(action)]))
    raise RuntimeError("expert planner could not find a DoorKey solution")


def run_episode(
    policy: str,
    size: int,
    seed: int,
    *,
    render_mode: str | None = None,
    capture_trace: bool = False,
) -> tuple[EpisodeResult, dict[str, list[Any]] | None]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    env = make_env(size, render_mode=render_mode)
    observation, _ = env.reset(seed=seed)
    del observation
    rng = np.random.default_rng(seed)
    plan = expert_plan(env) if policy == "expert" else []
    episode_return = 0.0
    picked_key = opened_door = False
    terminated = truncated = False
    steps = 0
    trace = (
        {"x": [], "y": [], "direction": [], "action": [], "reward": []} if capture_trace else None
    )
    try:
        while not (terminated or truncated):
            if policy == "expert":
                action = plan[steps]
            elif policy == "forward":
                action = int(Actions.forward)
            else:
                action = int(rng.integers(env.action_space.n))
            if trace is not None:
                trace["x"].append(int(env.unwrapped.agent_pos[0]))
                trace["y"].append(int(env.unwrapped.agent_pos[1]))
                trace["direction"].append(int(env.unwrapped.agent_dir))
                trace["action"].append(action)
            _, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            steps += 1
            picked_key = picked_key or isinstance(env.unwrapped.carrying, Key)
            _, door_position, _ = locate_objects_after_pickup(env)
            door = env.unwrapped.grid.get(*door_position)
            opened_door = opened_door or bool(door.is_open)
            if trace is not None:
                trace["reward"].append(float(reward))
    finally:
        env.close()
    return (
        EpisodeResult(
            policy=policy,
            size=size,
            seed=seed,
            success=bool(terminated and episode_return > 0),
            episode_return=episode_return,
            steps=steps,
            picked_key=picked_key,
            opened_door=opened_door,
            timeout=bool(truncated),
        ),
        trace,
    )


def locate_objects_after_pickup(
    env: gym.Env,
) -> tuple[tuple[int, int] | None, tuple[int, int], tuple[int, int]]:
    """Locate remaining objects after the key may have left the grid."""
    base = env.unwrapped
    key_position = door_position = goal_position = None
    for x in range(base.width):
        for y in range(base.height):
            cell = base.grid.get(x, y)
            if isinstance(cell, Key):
                key_position = (x, y)
            elif isinstance(cell, Door):
                door_position = (x, y)
            elif isinstance(cell, Goal):
                goal_position = (x, y)
    if door_position is None or goal_position is None:
        raise RuntimeError("DoorKey grid is missing a door or goal")
    return key_position, door_position, goal_position


Policy = Callable[[np.ndarray], int]

"""Gymnasium environment for the low-compute Snake lesson."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class SnakeEnv(gym.Env[np.ndarray, int]):
    """A compact Snake environment with relative left/right/forward actions."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human"],
        "render_fps": 30,
    }
    direction_channels: ClassVar[dict[tuple[int, int], int]] = {
        (-1, 0): 3,
        (1, 0): 4,
        (0, -1): 5,
        (0, 1): 6,
    }

    def __init__(
        self,
        grid_size: int = 10,
        end_score: int = 10,
        max_steps: int = 500,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        if end_score < 2:
            raise ValueError("end_score must be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'")

        self.grid_size = grid_size
        self.end_score = end_score
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(7, grid_size, grid_size),
            dtype=np.float32,
        )

        self.snake: list[np.ndarray] = []
        self.food_pos = np.zeros(2, dtype=np.int64)
        self.current_direction = np.array([0, 1], dtype=np.int64)
        self.score = 0
        self.step_count = 0
        self.visited: set[tuple[int, int]] = set()
        self.visited_positions: list[np.ndarray] = []
        self.game_over = False
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.snake = [self._get_center_position()]
        self.food_pos = self._generate_food()
        self.current_direction = np.array([0, 1], dtype=np.int64)
        self.score = 0
        self.step_count = 0
        self.visited.clear()
        self.visited_positions.clear()
        self.game_over = False
        self.closed = False
        observation = self._get_observation()
        if self.render_mode == "human":
            self.render()
        return observation, self._info(executed_action=None)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.game_over:
            raise RuntimeError("episode is finished; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}; expected 0, 1, or 2")

        executed_action = int(action)
        self.current_direction = self._get_direction(executed_action)
        new_head = self.snake[0] + self.current_direction
        self.step_count += 1

        if self._is_collision(new_head):
            reward = -5.0
            terminated = True
        else:
            reward = self._calculate_reward(new_head)
            reward_delta, terminated = self._update_snake_and_food(new_head)
            reward += reward_delta

        truncated = self.step_count >= self.max_steps and not terminated
        self.game_over = terminated or truncated
        observation = self._get_observation()
        info = self._info(executed_action=executed_action)
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def teacher_action(self) -> int:
        """Return a deterministic safe action for optional behavior cloning.

        The teacher never changes actions inside :meth:`step`. Training code must
        explicitly request this action and pass the same action to the environment.
        """

        candidates: list[tuple[float, int]] = []
        for action in range(self.action_space.n):
            trial_head = self.snake[0] + self._get_direction(action)
            if self._is_safe(trial_head) and self._bfs_safe_path(trial_head):
                distance = float(np.abs(trial_head - self.food_pos).sum())
                candidates.append((distance, action))
        if not candidates:
            return 2
        return min(candidates)[1]

    def _info(self, executed_action: int | None) -> dict[str, Any]:
        return {
            "score": self.score,
            "length": len(self.snake),
            "steps": self.step_count,
            "executed_action": executed_action,
        }

    def _is_safe(self, position: np.ndarray) -> bool:
        if np.any(position < 0) or np.any(position >= self.grid_size):
            return False
        return tuple(position) not in map(tuple, self.snake[1:])

    def _bfs_safe_path(self, position: np.ndarray, depth: int = 11) -> bool:
        directions = (
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, -1]),
            np.array([0, 1]),
        )
        queue: deque[np.ndarray] = deque([position])
        blocked = set(map(tuple, self.snake[1:]))
        visited = {tuple(position)}
        required_cells = min(depth, self.grid_size * self.grid_size - len(blocked))
        while queue:
            current = queue.popleft()
            if len(visited) >= required_cells:
                return True
            for direction in directions:
                candidate = current + direction
                candidate_key = tuple(candidate)
                in_bounds = np.all(candidate >= 0) and np.all(candidate < self.grid_size)
                if in_bounds and candidate_key not in blocked and candidate_key not in visited:
                    queue.append(candidate)
                    visited.add(candidate_key)
        return False

    def _get_direction(self, action: int) -> np.ndarray:
        if action == 0:
            return np.array([-self.current_direction[1], self.current_direction[0]])
        if action == 1:
            return np.array([self.current_direction[1], -self.current_direction[0]])
        return self.current_direction.copy()

    def _calculate_reward(self, new_head: np.ndarray) -> float:
        distance_before = np.abs(self.snake[0] - self.food_pos).sum()
        distance_after = np.abs(new_head - self.food_pos).sum()
        reward = 0.2 if distance_after < distance_before else -0.1

        if tuple(new_head) not in self.visited:
            reward += 0.2
            self.visited.add(tuple(new_head))
        if self._check_repeated_visit():
            reward -= 0.4
        return reward

    def _update_snake_and_food(self, new_head: np.ndarray) -> tuple[float, bool]:
        self.snake.insert(0, new_head)
        reward = 0.0
        if np.array_equal(new_head, self.food_pos):
            self.score += 1
            self.food_pos = self._generate_food()
            reward += 10.0
        else:
            self.snake.pop()
            reward -= 0.01

        self.visited_positions.append(new_head.copy())
        terminated = len(self.snake) >= self.end_score
        if terminated:
            reward += 100.0
        return reward, terminated

    def _check_repeated_visit(self) -> bool:
        if len(self.visited_positions) < 5:
            return False
        counts = Counter(map(tuple, self.visited_positions[-5:]))
        return any(count >= 2 for count in counts.values())

    def _get_observation(self) -> np.ndarray:
        grid = np.zeros((7, self.grid_size, self.grid_size), dtype=np.float32)
        for segment in self.snake:
            grid[0, tuple(segment)] = 1.0
        grid[1, tuple(self.food_pos)] = 1.0
        grid[2, :, :] = 1.0
        grid[2, 1:-1, 1:-1] = 0.0
        direction_channel = self.direction_channels[tuple(self.current_direction)]
        grid[direction_channel, :, :] = 1.0
        return grid

    def _is_collision(self, position: np.ndarray) -> bool:
        if np.any(position < 0) or np.any(position >= self.grid_size):
            return True
        return tuple(position) in map(tuple, self.snake[1:])

    def _generate_food(self) -> np.ndarray:
        occupied = set(map(tuple, self.snake))
        free_cells = [
            (row, column)
            for row in range(self.grid_size)
            for column in range(self.grid_size)
            if (row, column) not in occupied
        ]
        index = int(self.np_random.integers(len(free_cells)))
        return np.array(free_cells[index], dtype=np.int64)

    def _get_center_position(self) -> np.ndarray:
        return np.array([self.grid_size // 2, self.grid_size // 2], dtype=np.int64)

    def render(self, fps: int | None = None) -> None:
        if self.render_mode != "human":
            return
        if self.window is None:
            pygame.init()
            self.cell_size = 20
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
            pygame.display.set_caption("RL Snake")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                self.close()
                return

        self.window.fill((18, 18, 18))
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                rect = pygame.Rect(
                    column * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.window, (48, 48, 48), rect, 1)

        food_row, food_column = self.food_pos
        pygame.draw.rect(
            self.window,
            (62, 201, 111),
            pygame.Rect(
                food_column * self.cell_size,
                food_row * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )
        for index, (row, column) in enumerate(self.snake):
            color = (235, 87, 87) if index == 0 else (224, 132, 79)
            pygame.draw.rect(
                self.window,
                color,
                pygame.Rect(
                    column * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )
        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(fps or self.metadata["render_fps"])

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()

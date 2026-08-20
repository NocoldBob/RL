from __future__ import annotations

import numpy as np
import pytest
from environment import SnakeEnv
from gymnasium.utils.env_checker import check_env


def test_environment_passes_gymnasium_checker() -> None:
    check_env(SnakeEnv(grid_size=6, end_score=4, max_steps=20), skip_render_check=True)


def test_seed_reproduces_initial_observation() -> None:
    first = SnakeEnv()
    second = SnakeEnv()
    first_observation, _ = first.reset(seed=123)
    second_observation, _ = second.reset(seed=123)
    np.testing.assert_array_equal(first_observation, second_observation)
    np.testing.assert_array_equal(first.food_pos, second.food_pos)


def test_step_executes_the_requested_action_without_teacher_override() -> None:
    env = SnakeEnv(grid_size=8)
    observation, _ = env.reset(seed=7)
    assert observation.shape == (7, 8, 8)
    assert np.all(observation[6] == 1)
    original_head = env.snake[0].copy()
    next_observation, _, _, _, info = env.step(0)
    np.testing.assert_array_equal(env.snake[0], original_head + np.array([-1, 0]))
    assert np.all(next_observation[3] == 1)
    assert info["executed_action"] == 0


def test_episode_truncates_at_step_limit() -> None:
    env = SnakeEnv(grid_size=8, max_steps=1)
    env.reset(seed=7)
    _, _, terminated, truncated, _ = env.step(2)
    assert not terminated
    assert truncated
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(2)


@pytest.mark.parametrize("seed", range(5))
def test_teacher_reaches_food_without_overriding_step(seed: int) -> None:
    env = SnakeEnv(grid_size=6, end_score=4, max_steps=100)
    env.reset(seed=seed)
    for _ in range(100):
        action = env.teacher_action()
        _, _, terminated, truncated, info = env.step(action)
        assert info["executed_action"] == action
        if info["score"] >= 1:
            break
        assert not terminated
        assert not truncated
    assert info["score"] >= 1

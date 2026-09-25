from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from benchmark_lunar_baselines import build_report
from lunar_baselines import ENV_ID, POLICY_NAMES, engine_power, run_episode, select_action


def test_lunar_environment_has_two_continuous_actions() -> None:
    env = gym.make(ENV_ID)
    try:
        assert env.observation_space.shape == (8,)
        assert env.action_space.shape == (2,)
    finally:
        env.close()


def test_all_lunar_baselines_return_bounded_actions() -> None:
    env = gym.make(ENV_ID)
    try:
        observation, _ = env.reset(seed=7)
        for policy in POLICY_NAMES:
            action = select_action(policy, env, observation, np.random.default_rng(7))
            assert action.shape == (2,)
            assert action.dtype == np.float32
            assert np.all(action >= -1.0)
            assert np.all(action <= 1.0)
    finally:
        env.close()


def test_engine_dead_zones_match_environment_rules() -> None:
    assert engine_power(np.asarray([-1.0, 0.49], dtype=np.float32)) == (0.0, 0.0)
    assert engine_power(np.asarray([0.0, -0.5], dtype=np.float32)) == (0.0, 0.0)
    main, side = engine_power(np.asarray([0.4, -0.75], dtype=np.float32))
    assert main == pytest.approx(0.7)
    assert side == pytest.approx(0.75)


def test_lunar_seed_reproduces_random_episode() -> None:
    first, _ = run_episode("random", 123)
    second, _ = run_episode("random", 123)
    assert first.episode_return == pytest.approx(second.episode_return)
    assert first.steps == second.steps
    assert first.final_x == pytest.approx(second.final_x)


def test_small_lunar_benchmark_contains_all_policies() -> None:
    report = build_report(episodes=2, base_seed=11, trace_seed=42)
    assert set(report["aggregate"]) == set(POLICY_NAMES)
    assert len(report["episodes"]) == 2 * len(POLICY_NAMES)
    assert set(report["traces"]) == set(POLICY_NAMES)

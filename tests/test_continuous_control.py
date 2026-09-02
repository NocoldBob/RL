from __future__ import annotations

import numpy as np
import pytest
from benchmark_baselines import build_report
from mountain_car_baselines import POLICY_NAMES, run_episode, select_action


def test_all_baselines_return_bounded_continuous_actions() -> None:
    observation = np.asarray([-0.5, 0.02], dtype=np.float32)
    for policy in POLICY_NAMES:
        action = select_action(policy, observation, np.random.default_rng(7))
        assert action.shape == (1,)
        assert action.dtype == np.float32
        assert -1.0 <= action[0] <= 1.0


def test_seed_reproduces_random_baseline_episode() -> None:
    first = run_episode("random", 123)
    second = run_episode("random", 123)
    assert first.episode_return == pytest.approx(second.episode_return)
    assert first.steps == second.steps
    assert first.final_position == pytest.approx(second.final_position)


def test_zero_throttle_never_spends_action_energy() -> None:
    result = run_episode("zero", 7)
    assert not result.success
    assert result.action_energy == pytest.approx(0.0)
    assert result.episode_return == pytest.approx(0.0)


@pytest.mark.parametrize("policy", ["bang_bang", "smooth_momentum"])
def test_momentum_baselines_reach_the_goal(policy: str) -> None:
    result = run_episode(policy, 42)
    assert result.success
    assert result.episode_return > 0.0
    assert result.steps < 300


def test_small_benchmark_has_all_policies_and_episode_rows() -> None:
    report = build_report(episodes=2, base_seed=11, trace_seed=42)
    assert set(report["aggregate"]) == set(POLICY_NAMES)
    assert len(report["episodes"]) == 2 * len(POLICY_NAMES)
    assert set(report["traces"]) == set(POLICY_NAMES)

from __future__ import annotations

import numpy as np
import pytest
from inspect_decisions import (
    build_scenarios,
    centered_advantages,
    scenario_immediate_safety,
    scenario_observation,
)


def test_fixed_scenarios_produce_valid_observations() -> None:
    scenarios = build_scenarios()
    assert len(scenarios) == 4
    for scenario in scenarios:
        observation = scenario_observation(scenario)
        assert observation.shape == (7, 6, 6)
        assert observation.dtype == np.float32
        assert len(scenario_immediate_safety(scenario)) == 3


def test_fixed_scenarios_lock_the_intended_immediate_hazards() -> None:
    scenarios = {scenario.name: scenario for scenario in build_scenarios()}
    assert scenario_immediate_safety(scenarios["food_ahead"]) == (True, True, True)
    assert scenario_immediate_safety(scenarios["wall_ahead"]) == (True, True, False)
    assert scenario_immediate_safety(scenarios["narrow_escape"]) == (True, False, True)


def test_centered_advantages_have_zero_mean() -> None:
    centered = centered_advantages(np.asarray([2.0, -1.0, 5.0]))
    assert centered.mean() == pytest.approx(0.0)

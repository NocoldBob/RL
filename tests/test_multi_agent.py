from __future__ import annotations

import numpy as np
import torch
from ppo_agents import MultiAgentPPO, PPOConfig
from spread_env import SpreadConfig, assignment, coverage_metrics, make_env, privileged_actions
from value_agents import (
    GLOBAL_STATE_SIZE,
    LOCAL_OBSERVATION_SIZE,
    MultiAgentValueLearner,
    QMixer,
    ValueConfig,
    padded_observation,
    padded_state,
)


def test_spread_shapes_and_metrics() -> None:
    environment = make_env(SpreadConfig(agents=3, max_cycles=2))
    try:
        observations, _ = environment.reset(seed=5)
        assert set(observations) == {"agent_0", "agent_1", "agent_2"}
        assert all(value.shape == (18,) for value in observations.values())
        assert environment.state().shape == (54,)
        actions = privileged_actions(environment, coordinated=True)
        assert all(0 <= action < 5 for action in actions.values())
        assert 0 <= coverage_metrics(environment)["coverage"] <= 1
    finally:
        environment.close()


def test_assignment_finds_minimum_pairing() -> None:
    agents = np.asarray([[0, 0], [2, 0]], dtype=np.float32)
    landmarks = np.asarray([[2.1, 0], [0.1, 0]], dtype=np.float32)
    assert assignment(agents, landmarks) == (1, 0)


def test_padding_supports_variable_team_sizes() -> None:
    assert padded_observation(np.ones(12, dtype=np.float32)).shape == (LOCAL_OBSERVATION_SIZE,)
    assert padded_observation(np.ones(18, dtype=np.float32), agent_id=2).shape == (28,)
    assert padded_state(np.ones(54, dtype=np.float32)).shape == (GLOBAL_STATE_SIZE,)


def test_value_agents_and_monotonic_mixer_shapes() -> None:
    config = ValueConfig(mode="qmix", agent_id=True, total_steps=1)
    learner = MultiAgentValueLearner(config)
    observations = torch.zeros(4, 3, 28)
    assert learner._agent_q(observations).shape == (4, 3, 5)
    mixer = QMixer(3)
    values = torch.randn(4, 3, requires_grad=True)
    total = mixer(values, torch.zeros(4, GLOBAL_STATE_SIZE))
    total.sum().backward()
    assert total.shape == (4,)
    assert torch.all(values.grad >= 0)


def test_ippo_mappo_and_communication_actions() -> None:
    observations = np.zeros((3, LOCAL_OBSERVATION_SIZE), dtype=np.float32)
    state = np.zeros(GLOBAL_STATE_SIZE, dtype=np.float32)
    for mode in ("ippo", "mappo", "communication"):
        agent = MultiAgentPPO(PPOConfig(mode=mode, total_steps=1))
        actions, log_probabilities, values = agent.act(observations, state)
        assert actions.shape == (3,)
        assert log_probabilities.shape == (3,)
        assert values.shape == (3,)

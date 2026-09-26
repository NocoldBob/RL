from __future__ import annotations

import numpy as np
import torch
from carracing_dqn import VisualDQNAgent, VisualReplayBuffer
from carracing_env import (
    DISCRETE_ACTIONS,
    DrivingConfig,
    VisualCarRacing,
    continuous_action_from_policy,
    corrupt_frame,
    preprocess_frame,
)
from carracing_ppo import VisualPPOAgent, generalized_advantage_estimate


def test_preprocessing_and_corruptions() -> None:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    frame[:84, :, 1] = 255
    gray = preprocess_frame(frame, size=64, grayscale=True)
    rgb = preprocess_frame(frame, size=32, grayscale=False)
    assert gray.shape == (1, 64, 64)
    assert rgb.shape == (3, 32, 32)
    assert gray.dtype == np.float32
    rng = np.random.default_rng(7)
    for kind in ("brightness", "noise", "color", "occlusion", "blur"):
        changed = corrupt_frame(frame, kind, 0.5, rng)
        assert changed.shape == frame.shape
        assert changed.dtype == np.uint8


def test_action_mapping_and_discrete_table() -> None:
    mapped = continuous_action_from_policy(np.asarray([-2.0, 0.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(mapped, [-1.0, 0.5, 1.0])
    assert DISCRETE_ACTIONS.shape == (9, 3)
    assert np.all(DISCRETE_ACTIONS[:, 0] >= -1)
    assert np.all(DISCRETE_ACTIONS[:, 1:] >= 0)


def test_environment_shapes_and_short_truncation() -> None:
    environment = VisualCarRacing(
        DrivingConfig(frame_stack=2, frame_skip=1, action_mode="discrete", max_decisions=2)
    )
    try:
        observation, _ = environment.reset(seed=3)
        assert observation.shape == (2, 64, 64)
        _, _, _, truncated, _ = environment.step(0)
        assert not truncated
        next_observation, _, _, truncated, info = environment.step(0)
        assert next_observation.shape == observation.shape
        assert truncated
        assert info["env_action"].shape == (3,)
    finally:
        environment.close()


def test_visual_agents_and_replay_shapes() -> None:
    observation = np.zeros((4, 64, 64), dtype=np.float32)
    dqn = VisualDQNAgent(4, 9)
    assert dqn.online(torch.zeros(2, 4, 64, 64)).shape == (2, 9)
    assert 0 <= dqn.act(observation) < 9
    replay = VisualReplayBuffer(8, observation.shape)
    replay.append(observation, 2, 1.5, observation, False)
    batch = replay.sample(1, dqn.device)
    assert batch[0].shape == (1, 4, 64, 64)

    ppo = VisualPPOAgent(4)
    action, log_probability, value = ppo.act(observation)
    assert action.shape == (3,)
    assert np.isfinite([log_probability, value]).all()


def test_gae_respects_episode_boundary() -> None:
    advantages, returns = generalized_advantage_estimate(
        rewards=np.asarray([1.0, 2.0], dtype=np.float32),
        values=np.asarray([0.0, 0.0], dtype=np.float32),
        next_values=np.asarray([10.0, 10.0], dtype=np.float32),
        terminated=np.asarray([0.0, 1.0], dtype=np.float32),
        done=np.asarray([1.0, 1.0], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(advantages, [10.0, 2.0])
    np.testing.assert_allclose(returns, advantages)

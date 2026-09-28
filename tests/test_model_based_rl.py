from __future__ import annotations

import numpy as np
import pytest
import torch
from dynamics_model import (
    DynamicsDataset,
    DynamicsModel,
    ModelEnsemble,
    MPCController,
    OracleDynamics,
)
from model_based_sac import Replay, SimpleSAC
from pendulum_env import make_env


def dataset() -> DynamicsDataset:
    observations = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (8, 1))
    return DynamicsDataset(
        observations=observations,
        actions=np.zeros((8, 1), dtype=np.float32),
        rewards=np.zeros(8, dtype=np.float32),
        next_observations=observations.copy(),
        episode_id=np.repeat(np.arange(4), 2),
    )


def test_dynamics_dataset_split_keeps_episodes_separate() -> None:
    training, validation = dataset().split_by_episode(validation_fraction=0.25)
    assert set(training.episode_id).isdisjoint(set(validation.episode_id))
    assert len(training) + len(validation) == len(dataset())


def test_oracle_matches_gymnasium_one_step() -> None:
    env = make_env()
    observation, _ = env.reset(seed=7)
    action = np.asarray([0.75], dtype=np.float32)
    next_observation, reward, _, _, _ = env.step(action)
    env.close()
    predicted, predicted_reward = OracleDynamics().predict(
        torch.as_tensor(observation).reshape(1, 3), torch.as_tensor(action).reshape(1, 1)
    )
    np.testing.assert_allclose(predicted.numpy()[0], next_observation, atol=1e-6)
    assert float(predicted_reward[0]) == pytest.approx(reward, abs=1e-6)


def test_learned_model_and_ensemble_shapes() -> None:
    observations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])
    actions = torch.zeros(2, 1)
    models = [DynamicsModel(), DynamicsModel()]
    next_observations, rewards, uncertainty = ModelEnsemble(models).predict(observations, actions)
    assert next_observations.shape == (2, 3)
    assert rewards.shape == uncertainty.shape == (2,)
    assert torch.allclose(next_observations[:, :2].norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.all(uncertainty >= 0)


def test_planner_and_sac_smoke() -> None:
    controller = MPCController(OracleDynamics(), horizon=3, candidates=16, seed=1)
    action = controller(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    assert action.shape == (1,)
    assert -2.0 <= action[0] <= 2.0
    replay = Replay(seed=1)
    for _ in range(32):
        replay.append(np.asarray([1.0, 0.0, 0.0]), np.zeros(1), 0.0, np.asarray([1.0, 0.0, 0.0]))
    metrics = SimpleSAC(seed=1).update(replay.sample(16))
    assert all(np.isfinite(value) for value in metrics.values())

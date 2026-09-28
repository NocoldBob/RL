from __future__ import annotations

import numpy as np
import pytest
import torch
from dataset import OfflineDataset, concatenate
from off_policy_evaluation import importance_sampling
from offline_agents import OfflineAgent


def make_dataset(episodes: int = 4, steps: int = 3) -> OfflineDataset:
    size = episodes * steps
    episode_id = np.repeat(np.arange(episodes), steps)
    terminated = np.zeros(size, dtype=bool)
    terminated[steps - 1 :: steps] = True
    return OfflineDataset(
        observations=np.zeros((size, 6), dtype=np.float32),
        actions=np.arange(size, dtype=np.int64) % 3,
        rewards=-np.ones(size, dtype=np.float32),
        next_observations=np.ones((size, 6), dtype=np.float32),
        terminated=terminated,
        truncated=np.zeros(size, dtype=bool),
        episode_id=episode_id,
        behavior_probability=np.full(size, 1 / 3, dtype=np.float32),
        source=np.full(size, "test"),
    )


def test_episode_split_has_no_leakage() -> None:
    training, validation = make_dataset().split_by_episode(validation_fraction=0.25)
    assert set(training.episode_id).isdisjoint(set(validation.episode_id))
    assert len(training) + len(validation) == 12


def test_concatenate_offsets_episode_ids() -> None:
    combined = concatenate([make_dataset(2), make_dataset(2)])
    assert len(np.unique(combined.episode_id)) == 4
    assert len(combined) == 12


def test_all_offline_agents_can_update() -> None:
    dataset = make_dataset(8, 4)
    for mode in ("bc", "dqn", "cql", "iql"):
        agent = OfflineAgent(mode)
        metrics = agent.update(dataset.sample(16, torch.device("cpu")))
        assert metrics
        assert all(np.isfinite(value) for value in metrics.values())
        assert 0 <= agent.act(np.zeros(6, dtype=np.float32)) < 3


def test_importance_sampling_is_exact_for_matching_uniform_policy() -> None:
    dataset = make_dataset(3, 2)
    report = importance_sampling(dataset, OfflineAgent("bc"), gamma=1.0, target_epsilon=1.0)
    assert report["ordinary_is"] == pytest.approx(-2.0)
    assert report["weighted_is"] == pytest.approx(-2.0)
    assert report["effective_sample_size"] == pytest.approx(3.0)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from continuous_ppo import (
    ContinuousPPOAgent,
    ContinuousRolloutBuffer,
    continuous_gae,
    normalize_observation,
)
from train_continuous_ppo import ContinuousPPOConfig, train_continuous_ppo


def test_observation_normalization_maps_declared_bounds() -> None:
    assert np.allclose(normalize_observation(np.asarray([-1.2, -0.07])), [-1.0, -1.0])
    assert np.allclose(normalize_observation(np.asarray([0.6, 0.07])), [1.0, 1.0])


def test_sampled_actions_are_bounded_and_log_probability_is_reproducible() -> None:
    torch.manual_seed(7)
    agent = ContinuousPPOAgent()
    observation = np.asarray([-0.5, 0.01], dtype=np.float32)
    action, sampled_log_probability, _ = agent.select_action(observation)
    state = torch.as_tensor(normalize_observation(observation)).unsqueeze(0)
    replayed_log_probability, _, _ = agent.model.evaluate_actions(
        state, torch.as_tensor(action).unsqueeze(0)
    )
    assert action.shape == (1,)
    assert -1.0 <= action[0] <= 1.0
    assert np.isfinite(sampled_log_probability)
    assert float(replayed_log_probability.item()) == pytest.approx(
        sampled_log_probability, abs=1e-5
    )


def test_truncation_bootstraps_but_does_not_cross_episode_boundary() -> None:
    advantages, returns = continuous_gae(
        rewards=np.asarray([1.0, 10.0], dtype=np.float32),
        values=np.asarray([2.0, 4.0], dtype=np.float32),
        next_values=np.asarray([3.0, 0.0], dtype=np.float32),
        terminated=np.asarray([0.0, 1.0], dtype=np.float32),
        episode_done=np.asarray([1.0, 1.0], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.95,
    )
    assert advantages[0] == pytest.approx(1.0 + 0.9 * 3.0 - 2.0)
    assert advantages[1] == pytest.approx(10.0 - 4.0)
    assert returns[0] == pytest.approx(3.7)


def test_update_changes_model_parameters() -> None:
    torch.manual_seed(11)
    agent = ContinuousPPOAgent()
    buffer = ContinuousRolloutBuffer()
    for index in range(32):
        observation = np.asarray([-0.5 + index * 0.001, 0.01], dtype=np.float32)
        action, log_probability, value = agent.select_action(observation)
        buffer.append(
            normalize_observation(observation),
            action,
            0.1,
            False,
            index == 31,
            log_probability,
            value,
            value,
        )
    before = [parameter.detach().clone() for parameter in agent.model.parameters()]
    agent.update(
        buffer.build_batch(torch.device("cpu"), gamma=0.99, gae_lambda=0.95),
        update_epochs=2,
        batch_size=16,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
    )
    assert any(
        not torch.equal(old, new) for old, new in zip(before, agent.model.parameters(), strict=True)
    )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    agent = ContinuousPPOAgent()
    observation = np.asarray([-0.5, 0.01], dtype=np.float32)
    expected = agent.predict(observation)
    path = tmp_path / "agent.pt"
    agent.save_checkpoint(path, steps=10, config={"example": True})
    restored = ContinuousPPOAgent()
    payload = restored.load_checkpoint(path)
    assert payload["steps"] == 10
    assert np.allclose(restored.predict(observation), expected)


def test_short_training_writes_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "run"
    summary = train_continuous_ppo(
        ContinuousPPOConfig(
            total_steps=64,
            rollout_steps=32,
            update_epochs=1,
            batch_size=16,
            eval_interval=64,
            eval_episodes=1,
            output_dir=output,
        )
    )
    assert summary["total_steps"] == 64
    assert (output / "checkpoints" / "best.pt").is_file()
    assert (output / "checkpoints" / "latest.pt").is_file()
    assert (output / "history.json").is_file()

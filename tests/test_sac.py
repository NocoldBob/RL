from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sac import ReplayBuffer, SACAgent, normalize_observation
from train_sac import SACConfig, train_sac


def populate_replay(buffer: ReplayBuffer, count: int) -> None:
    for index in range(count):
        state = np.asarray([-0.5 + index * 0.001, 0.01], dtype=np.float32)
        buffer.append(
            normalize_observation(state),
            np.asarray([0.25], dtype=np.float32),
            0.1,
            normalize_observation(state + np.asarray([0.001, 0.0], dtype=np.float32)),
            False,
        )


def test_replay_buffer_wraps_and_samples_expected_shapes() -> None:
    buffer = ReplayBuffer(8, 2, 1, seed=7)
    populate_replay(buffer, 10)
    batch = buffer.sample(4, torch.device("cpu"))
    assert len(buffer) == 8
    assert batch.states.shape == (4, 2)
    assert batch.actions.shape == (4, 1)
    assert batch.rewards.shape == (4, 1)
    assert batch.terminated.shape == (4, 1)


def test_sac_actions_are_bounded_and_deterministic_prediction_repeats() -> None:
    torch.manual_seed(7)
    agent = SACAgent()
    observation = np.asarray([-0.5, 0.01], dtype=np.float32)
    first = agent.predict(observation)
    second = agent.predict(observation)
    stochastic = agent.predict(observation, deterministic=False)
    assert np.allclose(first, second)
    assert first.shape == (1,)
    assert -1.0 <= first[0] <= 1.0
    assert -1.0 <= stochastic[0] <= 1.0


def test_sac_update_changes_networks_and_alpha() -> None:
    torch.manual_seed(11)
    buffer = ReplayBuffer(64, 2, 1, seed=11)
    populate_replay(buffer, 64)
    agent = SACAgent()
    actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    q_before = [parameter.detach().clone() for parameter in agent.q1.parameters()]
    alpha_before = float(agent.alpha.detach())
    metrics = agent.update(buffer.sample(32, agent.device))
    assert any(
        not torch.equal(old, new)
        for old, new in zip(actor_before, agent.actor.parameters(), strict=True)
    )
    assert any(
        not torch.equal(old, new) for old, new in zip(q_before, agent.q1.parameters(), strict=True)
    )
    assert float(agent.alpha.detach()) != pytest.approx(alpha_before)
    assert set(metrics) == {
        "q_loss",
        "actor_loss",
        "alpha_loss",
        "alpha",
        "q1_mean",
        "q2_mean",
        "target_q_mean",
    }


def test_sac_checkpoint_round_trip(tmp_path: Path) -> None:
    agent = SACAgent()
    observation = np.asarray([-0.5, 0.01], dtype=np.float32)
    expected = agent.predict(observation)
    path = tmp_path / "sac.pt"
    agent.save_checkpoint(path, steps=12, config={"example": True})
    restored = SACAgent()
    payload = restored.load_checkpoint(path)
    assert payload["steps"] == 12
    assert np.allclose(restored.predict(observation), expected)


def test_short_sac_training_writes_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "run"
    summary = train_sac(
        SACConfig(
            total_steps=64,
            learning_starts=16,
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

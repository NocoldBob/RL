from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ppo import PPOAgent, RolloutBuffer, generalized_advantage_estimate
from train_ppo import PPOTrainConfig, train_ppo


def make_rollout(count: int = 8) -> RolloutBuffer:
    rollout = RolloutBuffer()
    agent = PPOAgent(7, 3, 5)
    for index in range(count):
        state = np.zeros((7, 5, 5), dtype=np.float32)
        state[index % 7, index % 5, (index * 2) % 5] = 1.0
        action, log_probability, value = agent.select_action(state)
        rollout.append(
            state,
            action,
            reward=float(index % 3),
            done=index in {3, count - 1},
            log_probability=log_probability,
            value=value,
        )
    return rollout


def test_gae_stops_at_episode_boundaries() -> None:
    advantages, returns = generalized_advantage_estimate(
        rewards=np.array([1.0, 2.0, 10.0], dtype=np.float32),
        values=np.zeros(3, dtype=np.float32),
        dones=np.array([0.0, 1.0, 1.0], dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(advantages, [3.0, 2.0, 10.0])
    np.testing.assert_allclose(returns, advantages)


def test_rollout_batch_is_normalized_and_typed() -> None:
    batch = make_rollout().build_batch(
        torch.device("cpu"),
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert batch.states.shape == (8, 7, 5, 5)
    assert batch.actions.dtype == torch.int64
    assert batch.returns.dtype == torch.float32
    assert float(batch.advantages.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(batch.advantages.std(unbiased=False)) == pytest.approx(1.0, abs=1e-5)


def test_ppo_update_changes_parameters() -> None:
    torch.manual_seed(3)
    agent = PPOAgent(7, 3, 5)
    rollout = RolloutBuffer()
    for index in range(8):
        state = np.zeros((7, 5, 5), dtype=np.float32)
        state[index % 7, index % 5, (index * 2) % 5] = 1.0
        action, log_probability, value = agent.select_action(state)
        rollout.append(state, action, 1.0, index == 7, log_probability, value)
    before = [parameter.detach().clone() for parameter in agent.model.parameters()]
    metrics = agent.update(
        rollout.build_batch(torch.device("cpu"), gamma=0.99, gae_lambda=0.95),
        update_epochs=2,
        batch_size=4,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(expected, actual)
        for expected, actual in zip(before, agent.model.parameters(), strict=True)
    )


def test_stochastic_prediction_is_reproducible_with_generator() -> None:
    agent = PPOAgent(7, 3, 5)
    observation = np.zeros((7, 5, 5), dtype=np.float32)
    first = torch.Generator().manual_seed(17)
    second = torch.Generator().manual_seed(17)
    first_actions = [agent.predict(observation, generator=first) for _ in range(10)]
    second_actions = [agent.predict(observation, generator=second) for _ in range(10)]
    assert first_actions == second_actions


def test_ppo_checkpoint_round_trip(tmp_path: Path) -> None:
    agent = PPOAgent(7, 3, 5)
    checkpoint = tmp_path / "ppo.pt"
    agent.save_checkpoint(
        checkpoint,
        episode=12,
        total_steps=345,
        config={"grid_size": 5},
    )
    restored = PPOAgent(7, 3, 5)
    metadata = restored.load_checkpoint(checkpoint)
    assert metadata == {
        "episode": 12,
        "total_steps": 345,
        "config": {"grid_size": 5},
    }
    for expected, actual in zip(
        agent.model.parameters(),
        restored.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(expected, actual)


def test_short_ppo_training_creates_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "ppo"
    summary = train_ppo(
        PPOTrainConfig(
            episodes=4,
            grid_size=5,
            end_score=3,
            max_steps=6,
            rollout_steps=8,
            update_epochs=2,
            batch_size=4,
            eval_interval=2,
            eval_episodes=2,
            save_interval=2,
            output_dir=output_dir,
            tensorboard=False,
        )
    )
    assert summary["algorithm"] == "ppo"
    assert summary["updates"] >= 1
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert (output_dir / "history.json").is_file()

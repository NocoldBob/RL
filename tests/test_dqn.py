from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from benchmark_double_dqn import evaluate_q_diagnostics
from dqn import DQNAgent, ReplayBuffer
from train_dqn import DQNTrainConfig, epsilon_at_step, train_dqn


def fill_replay(buffer: ReplayBuffer, count: int = 8) -> None:
    for index in range(count):
        state = np.full((7, 5, 5), index, dtype=np.float32)
        buffer.append(state, index % 3, float(index), state + 1, index % 2 == 0)


def test_replay_buffer_returns_typed_batch() -> None:
    buffer = ReplayBuffer(capacity=10, seed=3)
    fill_replay(buffer)
    batch = buffer.sample(4, torch.device("cpu"))
    assert batch.states.shape == (4, 7, 5, 5)
    assert batch.actions.shape == (4,)
    assert batch.actions.dtype == torch.int64
    assert batch.dones.dtype == torch.float32


def test_dqn_update_and_target_sync() -> None:
    torch.manual_seed(3)
    agent = DQNAgent(7, 3, 5)
    buffer = ReplayBuffer(capacity=10, seed=3)
    fill_replay(buffer)
    target_before = [parameter.detach().clone() for parameter in agent.target.parameters()]
    online_before = [parameter.detach().clone() for parameter in agent.online.parameters()]
    metrics = agent.update(buffer.sample(4, torch.device("cpu")))
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(online_before, agent.online.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(target_before, agent.target.parameters(), strict=True)
    )
    agent.sync_target()
    for online, target in zip(agent.online.parameters(), agent.target.parameters(), strict=True):
        torch.testing.assert_close(online, target)


def test_double_dqn_separates_action_selection_from_evaluation() -> None:
    agent = DQNAgent(7, 3, 5)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        for parameter in agent.target.parameters():
            parameter.zero_()
        agent.online.network[-1].bias.copy_(torch.tensor([3.0, 1.0, 0.0]))
        agent.target.network[-1].bias.copy_(torch.tensor([1.0, 5.0, 0.0]))
    states = torch.zeros((2, 7, 5, 5))
    dqn_values = agent.bootstrap_values(states, double_dqn=False)
    double_dqn_values = agent.bootstrap_values(states, double_dqn=True)
    torch.testing.assert_close(dqn_values, torch.full((2,), 5.0))
    torch.testing.assert_close(double_dqn_values, torch.full((2,), 1.0))


def test_dqn_checkpoint_round_trip(tmp_path: Path) -> None:
    agent = DQNAgent(7, 3, 5)
    checkpoint = tmp_path / "dqn.pt"
    agent.save_checkpoint(
        checkpoint,
        episode=12,
        total_steps=345,
        config={"grid_size": 5},
    )
    restored = DQNAgent(7, 3, 5)
    metadata = restored.load_checkpoint(checkpoint)
    assert metadata == {
        "episode": 12,
        "total_steps": 345,
        "config": {"grid_size": 5},
    }
    for expected, actual in zip(
        agent.online.parameters(),
        restored.online.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(expected, actual)


def test_epsilon_schedule_and_greedy_action() -> None:
    config = DQNTrainConfig(epsilon_decay_steps=100)
    assert epsilon_at_step(config, 0) == 1.0
    assert epsilon_at_step(config, 100) == pytest.approx(0.05)
    assert epsilon_at_step(config, 1_000) == pytest.approx(0.05)
    agent = DQNAgent(7, 3, 5)
    observation = np.zeros((7, 5, 5), dtype=np.float32)
    action = agent.select_action(
        observation,
        epsilon=0.0,
        random_source=random.Random(2),
    )
    assert action in {0, 1, 2}


def test_short_dqn_training_creates_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "dqn"
    summary = train_dqn(
        DQNTrainConfig(
            episodes=4,
            grid_size=5,
            end_score=3,
            max_steps=6,
            replay_capacity=32,
            batch_size=4,
            learning_starts=4,
            train_interval=1,
            target_update_interval=3,
            eval_interval=2,
            eval_episodes=2,
            save_interval=2,
            output_dir=output_dir,
            tensorboard=False,
        )
    )
    assert summary["algorithm"] == "dqn"
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert (output_dir / "history.json").is_file()


def test_short_double_dqn_training_and_diagnostics(tmp_path: Path) -> None:
    output_dir = tmp_path / "double-dqn"
    summary = train_dqn(
        DQNTrainConfig(
            episodes=4,
            grid_size=5,
            end_score=3,
            max_steps=6,
            replay_capacity=32,
            batch_size=4,
            learning_starts=4,
            train_interval=1,
            target_update_interval=3,
            double_dqn=True,
            eval_interval=0,
            save_interval=0,
            output_dir=output_dir,
            tensorboard=False,
        )
    )
    assert summary["algorithm"] == "double_dqn"
    checkpoint = output_dir / "checkpoints" / "latest.pt"
    assert checkpoint.is_file()
    restored = DQNAgent(7, 3, 5)
    restored.load_checkpoint(checkpoint)
    diagnostics = evaluate_q_diagnostics(
        restored,
        grid_size=5,
        end_score=3,
        max_steps=6,
        episodes=3,
        seed_base=500,
        gamma=0.99,
    )
    assert all(np.isfinite(value) for value in diagnostics.values())

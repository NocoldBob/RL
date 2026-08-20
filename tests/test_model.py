from __future__ import annotations

from pathlib import Path

import torch
from model import ConvActorCritic


def make_model() -> ConvActorCritic:
    torch.manual_seed(3)
    return ConvActorCritic(3, 3, 6)


def test_actor_critic_update_changes_parameters() -> None:
    model = make_model()
    state = torch.zeros((1, 3, 6, 6))
    next_state = torch.ones((1, 3, 6, 6))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = model.update(state, 1, 1.0, next_state, False)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters(), strict=True)
    )


def test_imitation_update_changes_policy() -> None:
    model = make_model()
    state = torch.zeros((1, 3, 6, 6))
    before = model.actor.weight.detach().clone()
    loss = model.imitation_update(state, teacher_action=2)
    assert loss > 0
    assert not torch.equal(before, model.actor.weight)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = make_model()
    checkpoint = tmp_path / "model.pt"
    model.save_checkpoint(checkpoint, episode=12, config={"grid_size": 6})
    restored = make_model()
    metadata = restored.load_checkpoint(checkpoint)
    assert metadata == {"episode": 12, "config": {"grid_size": 6}}
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)

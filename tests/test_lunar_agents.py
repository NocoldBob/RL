from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lunar_agents import LunarPPOAgent, LunarSACAgent, normalize_lunar_observation
from train_lunar_ppo import LunarPPOConfig, train_lunar_ppo
from train_lunar_sac import LunarSACConfig, train_lunar_sac

OBSERVATION = np.asarray([0.0, 1.0, 0.1, -0.2, 0.05, 0.1, 0.0, 0.0], dtype=np.float32)


def test_lunar_normalization_and_agent_action_shapes() -> None:
    normalized = normalize_lunar_observation(OBSERVATION)
    assert normalized.shape == (8,)
    for agent in (LunarPPOAgent(), LunarSACAgent()):
        action = agent.predict(OBSERVATION)
        assert action.shape == (2,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)


def test_lunar_checkpoint_round_trips(tmp_path: Path) -> None:
    ppo = LunarPPOAgent()
    ppo_path = tmp_path / "ppo.pt"
    ppo.save_checkpoint(ppo_path, steps=3, config={})
    restored_ppo = LunarPPOAgent()
    restored_ppo.load_checkpoint(ppo_path)
    assert np.allclose(ppo.predict(OBSERVATION), restored_ppo.predict(OBSERVATION))

    sac = LunarSACAgent()
    sac_path = tmp_path / "sac.pt"
    sac.save_checkpoint(sac_path, steps=4, config={})
    restored_sac = LunarSACAgent()
    restored_sac.load_checkpoint(sac_path)
    assert np.allclose(sac.predict(OBSERVATION), restored_sac.predict(OBSERVATION))


def test_short_lunar_training_writes_artifacts(tmp_path: Path) -> None:
    ppo_output = tmp_path / "ppo"
    train_lunar_ppo(
        LunarPPOConfig(
            total_steps=32,
            rollout_steps=16,
            update_epochs=2,
            batch_size=8,
            target_kl=-1.0,
            eval_interval=32,
            eval_episodes=1,
            output_dir=ppo_output,
        )
    )
    assert (ppo_output / "checkpoints" / "best.pt").is_file()
    assert (ppo_output / "history.json").is_file()
    ppo_history = json.loads((ppo_output / "history.json").read_text(encoding="utf-8"))
    first_update = ppo_history["updates"][0]
    assert first_update["epochs_completed"] == 1.0
    assert first_update["early_stopped"] == 1.0
    assert "gradient_norm" in first_update
    assert "explained_variance" in first_update

    sac_output = tmp_path / "sac"
    train_lunar_sac(
        LunarSACConfig(
            total_steps=32,
            replay_capacity=64,
            learning_starts=8,
            batch_size=8,
            update_frequency=2,
            eval_interval=32,
            eval_episodes=1,
            output_dir=sac_output,
        )
    )
    assert (sac_output / "checkpoints" / "best.pt").is_file()
    assert (sac_output / "history.json").is_file()

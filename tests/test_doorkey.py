from __future__ import annotations

import numpy as np
import torch
from doorkey_dqn import DoorKeyDQNAgent
from doorkey_env import env_id, expert_plan, make_env, run_episode
from doorkey_ppo import DoorKeyPPOAgent


def test_environment_modes_and_ids() -> None:
    assert env_id(5) == "MiniGrid-DoorKey-5x5-v0"
    partial = make_env(5)
    full = make_env(5, observation_mode="compact_full")
    rgb = make_env(5, observation_mode="rgb_partial")
    try:
        partial_observation, _ = partial.reset(seed=1)
        full_observation, _ = full.reset(seed=1)
        rgb_observation, _ = rgb.reset(seed=1)
        assert partial_observation["image"].shape == (7, 7, 3)
        assert full_observation.shape == (5, 5, 3)
        assert rgb_observation.shape == (56, 56, 3)
    finally:
        partial.close()
        full.close()
        rgb.close()


def test_expert_plan_solves_multiple_layouts() -> None:
    for size in (5, 8):
        for seed in (0, 7, 42):
            result, trace = run_episode("expert", size, seed, capture_trace=True)
            assert result.success
            assert result.picked_key
            assert result.opened_door
            assert result.episode_return > 0
            assert trace is not None
            assert len(trace["action"]) == result.steps


def test_planner_returns_valid_discrete_actions() -> None:
    env = make_env(5)
    try:
        env.reset(seed=2026)
        plan = expert_plan(env)
        assert plan
        assert np.all(np.asarray(plan) >= 0)
        assert np.all(np.asarray(plan) < env.action_space.n)
    finally:
        env.close()


def test_dqn_and_ppo_network_shapes() -> None:
    env = make_env(5, observation_mode="compact_partial")
    try:
        observation, _ = env.reset(seed=42)
        image = torch.as_tensor(observation["image"]).unsqueeze(0)
        dqn = DoorKeyDQNAgent(7, 7)
        assert dqn.online(image).shape == (1, 7)
        ppo = DoorKeyPPOAgent(recurrent=True)
        action, log_probability, value, hidden = ppo.act(observation["image"], ppo.initial_hidden())
        assert 0 <= action < 7
        assert np.isfinite(log_probability)
        assert np.isfinite(value)
        assert hidden.shape == (1, 64)
    finally:
        env.close()

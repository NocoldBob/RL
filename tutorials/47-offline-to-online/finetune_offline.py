"""Compare direct, mixed-replay, and conservative offline-to-online fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from acrobot_env import evaluate_policy
from dataset import OfflineDataset
from offline_agents import OfflineAgent
from online_dqn import DQNAgent, ReplayBuffer


def load_cql_q(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload["q"]


def merged_dqn_batch(
    online: ReplayBuffer,
    offline: OfflineDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    online_batch = online.sample(batch_size // 2, device)
    offline_batch = offline.sample(batch_size - batch_size // 2, device)
    offline_tuple = (
        offline_batch["observations"],
        offline_batch["actions"],
        offline_batch["rewards"],
        offline_batch["next_observations"],
        offline_batch["terminated"],
    )
    return tuple(
        torch.cat([first, second], dim=0)
        for first, second in zip(online_batch, offline_tuple, strict=True)
    )


def merged_offline_batch(
    online: ReplayBuffer,
    offline: OfflineDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    online_tuple = online.sample(batch_size // 2, device)
    offline_batch = offline.sample(batch_size - batch_size // 2, device)
    keys = ("observations", "actions", "rewards", "next_observations", "terminated")
    result = {
        key: torch.cat([online_value, offline_batch[key]], dim=0)
        for key, online_value in zip(keys, online_tuple, strict=True)
    }
    result["truncated"] = torch.zeros(batch_size, device=device)
    result["behavior_probability"] = torch.ones(batch_size, device=device)
    return result


def train_variant(
    name: str,
    checkpoint: Path,
    offline: OfflineDataset,
    *,
    total_steps: int = 5_000,
    seed: int = 42,
) -> dict[str, Any]:
    environment = gym.make("Acrobot-v1")
    online = ReplayBuffer(capacity=20_000)
    q_state = load_cql_q(checkpoint)
    if name == "conservative":
        learner: DQNAgent | OfflineAgent = OfflineAgent("cql", cql_alpha=0.5)
        assert learner.q is not None and learner.target_q is not None
        learner.q.load_state_dict(q_state)
        learner.target_q.load_state_dict(q_state)
    else:
        learner = DQNAgent(learning_rate=2e-4)
        learner.online.load_state_dict(q_state)
        learner.target.load_state_dict(q_state)
    observation, _ = environment.reset(seed=seed)
    history = {"evaluations": [], "updates": []}
    best_return = -float("inf")
    output_dir = Path("runs/offline-finetune") / name
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for step in range(1, total_steps + 1):
            action = (
                learner.act(observation) if np.random.random() > 0.1 else int(np.random.randint(3))
            )
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            online.append(observation, action, reward, next_observation, terminated)
            observation = next_observation
            if terminated or truncated:
                observation, _ = environment.reset()
            if len(online) >= 256:
                if name == "direct":
                    assert isinstance(learner, DQNAgent)
                    metrics = learner.update(online.sample(256, learner.device), 0.99)
                elif name == "mixed_replay":
                    assert isinstance(learner, DQNAgent)
                    metrics = learner.update(
                        merged_dqn_batch(online, offline, 256, learner.device), 0.99
                    )
                else:
                    assert isinstance(learner, OfflineAgent)
                    metrics = learner.update(
                        merged_offline_batch(online, offline, 256, learner.device)
                    )
                if step % 100 == 0:
                    history["updates"].append({"step": step, **metrics})
            if step % 250 == 0:
                learner.sync_target()
            if step % 1_000 == 0:
                summary = evaluate_policy(learner.act, list(range(85_000, 85_020)))["summary"]
                history["evaluations"].append({"step": step, **summary})
                print(f"{name} step={step} return={summary['average_return']:.1f}")
                if summary["average_return"] > best_return:
                    best_return = summary["average_return"]
                    if isinstance(learner, DQNAgent):
                        learner.save(output_dir / "best.pt", step, {"variant": name})
                    else:
                        learner.save(output_dir / "best.pt", step, {"variant": name})
    finally:
        environment.close()
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"best_average_return": best_return, "history": history}


def main() -> None:
    torch.set_num_threads(1)
    np.random.seed(42)
    torch.manual_seed(42)
    checkpoint = Path("runs/offline/cql-mixed-alpha-1/checkpoints/best.pt")
    offline = OfflineDataset.load(Path("datasets/acrobot/mixed.npz"))
    results = {
        name: train_variant(name, checkpoint, offline)
        for name in ("direct", "mixed_replay", "conservative")
    }
    report = {"experiment": "47-offline-to-online", "online_budget": 5_000, "results": results}
    path = Path("docs/experiments/47-offline-to-online.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: row["best_average_return"] for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()

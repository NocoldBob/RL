"""Train a DoorKey policy from expert demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from doorkey_dqn import image_observation
from doorkey_env import expert_plan, make_env
from doorkey_ppo import DoorKeyPPOAgent, evaluate_ppo, seed_everything


def collect_demonstrations(size: int, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    env = make_env(size, observation_mode="compact_partial")
    observations = []
    actions = []
    try:
        for seed in seeds:
            raw_observation, _ = env.reset(seed=seed)
            plan = expert_plan(env)
            for action in plan:
                observations.append(image_observation(raw_observation).copy())
                actions.append(action)
                raw_observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()
    return np.stack(observations), np.asarray(actions, dtype=np.int64)


def classification_accuracy(
    agent: DoorKeyPPOAgent,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    batch_size: int = 512,
) -> float:
    correct = 0
    with torch.no_grad():
        for start in range(0, len(actions), batch_size):
            observation_batch = torch.as_tensor(
                observations[start : start + batch_size], device=agent.device
            )
            hidden = agent.initial_hidden(len(observation_batch))
            logits, _, _ = agent.model(observation_batch, hidden)
            predictions = logits.argmax(dim=1).cpu().numpy()
            correct += int((predictions == actions[start : start + batch_size]).sum())
    return correct / len(actions)


def train_imitation(
    *,
    size: int,
    demonstrations: int,
    epochs: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = int(demonstrations * 0.8)
    train_observations, train_actions = collect_demonstrations(size, list(range(split)))
    validation_observations, validation_actions = collect_demonstrations(
        size, list(range(split, demonstrations))
    )
    agent = DoorKeyPPOAgent(recurrent=False, learning_rate=3e-4)
    rng = np.random.default_rng(seed)
    history = []
    best_validation = -1.0
    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(train_actions))
        losses = []
        for start in range(0, len(indices), 256):
            selected = indices[start : start + 256]
            observations = torch.as_tensor(train_observations[selected], device=agent.device)
            actions = torch.as_tensor(train_actions[selected], device=agent.device)
            logits, _, _ = agent.model(observations, agent.initial_hidden(len(selected)))
            loss = F.cross_entropy(logits, actions)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            losses.append(float(loss.detach()))
        train_accuracy = classification_accuracy(agent, train_observations, train_actions)
        validation_accuracy = classification_accuracy(
            agent, validation_observations, validation_actions
        )
        evaluation = evaluate_ppo(agent, list(range(60_000, 60_100)), size=size)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            **evaluation,
        }
        history.append(row)
        print(
            f"BC epoch={epoch} validation={validation_accuracy:.1%} "
            f"success={evaluation['success_rate']:.0%}"
        )
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy
            agent.save(
                output_dir / "checkpoints" / "best.pt",
                steps=epoch,
                config={
                    "size": size,
                    "demonstrations": demonstrations,
                    "epochs": epochs,
                    "seed": seed,
                },
            )
    report = {
        "experiment": "lesson-23-behavior-cloning",
        "size": size,
        "demonstration_episodes": demonstrations,
        "training_samples": len(train_actions),
        "validation_samples": len(validation_actions),
        "history": history,
    }
    (output_dir / "history.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone the DoorKey expert")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--demonstrations", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/doorkey-imitation/bc-8x8"))
    args = parser.parse_args()
    report = train_imitation(
        size=args.size,
        demonstrations=args.demonstrations,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["history"][-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

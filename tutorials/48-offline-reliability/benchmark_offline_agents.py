"""Evaluate offline checkpoints only after their training and selection are complete."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from acrobot_env import evaluate_policy
from offline_agents import OfflineAgent


def load_agent(checkpoint: Path) -> tuple[OfflineAgent, int, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    agent = OfflineAgent(
        config["mode"],
        cql_alpha=config.get("cql_alpha", 1.0),
        expectile=config.get("expectile", 0.7),
        advantage_beta=config.get("advantage_beta", 3.0),
    )
    step = agent.load(checkpoint)
    return agent, step, config


def evaluate(checkpoint: Path, episodes: int = 100) -> dict[str, Any]:
    agent, step, config = load_agent(checkpoint)
    report = evaluate_policy(agent.act, list(range(60_000, 60_000 + episodes)))
    return {"checkpoint_step": step, "config": config, **report}


def save(number: int, name: str, results: dict[str, Any]) -> None:
    path = Path(f"docs/experiments/{number}-{name}.json")
    path.write_text(
        json.dumps(
            {"experiment": f"{number}-{name}", "results": results}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )


def main() -> None:
    torch.set_num_threads(1)
    root = Path("runs/offline")
    checkpoints = {
        "bc_expert": root / "bc-expert/checkpoints/best.pt",
        "bc_mixed": root / "bc-mixed/checkpoints/best.pt",
        "dqn_mixed": root / "dqn-mixed/checkpoints/best.pt",
        "cql_alpha_0.1": root / "cql-mixed-alpha-0.1/checkpoints/best.pt",
        "cql_alpha_1": root / "cql-mixed-alpha-1/checkpoints/best.pt",
        "cql_alpha_5": root / "cql-mixed-alpha-5/checkpoints/best.pt",
        "iql_mixed": root / "iql-mixed/checkpoints/best.pt",
        "cql_random": root / "cql-random/checkpoints/best.pt",
        "cql_medium": root / "cql-medium/checkpoints/best.pt",
        "cql_expert": root / "cql-expert/checkpoints/best.pt",
    }
    evaluated = {name: evaluate(path) for name, path in checkpoints.items()}
    save(41, "behavior-cloning", {name: evaluated[name] for name in ("bc_expert", "bc_mixed")})
    save(42, "offline-dqn", {"offline_dqn": evaluated["dqn_mixed"]})
    save(
        43,
        "cql",
        {
            name: evaluated[name]
            for name in ("dqn_mixed", "cql_alpha_0.1", "cql_alpha_1", "cql_alpha_5")
        },
    )
    save(44, "iql", {name: evaluated[name] for name in ("bc_mixed", "cql_alpha_1", "iql_mixed")})
    save(
        45,
        "dataset-quality",
        {
            name: evaluated[name]
            for name in ("cql_random", "cql_medium", "cql_expert", "cql_alpha_1")
        },
    )
    print(
        json.dumps(
            {name: row["summary"] for name, row in evaluated.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

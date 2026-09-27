"""Compile training histories into the chapter 33-37 experiment reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def variant(path: str) -> dict[str, Any]:
    history = load(path)
    return {
        "best": max(history["evaluations"], key=lambda row: row["average_team_return"]),
        "history": history,
    }


def save(number: int, name: str, variants: dict[str, Any]) -> None:
    path = Path(f"docs/experiments/{number}-{name}.json")
    path.write_text(
        json.dumps(
            {"experiment": f"{number}-{name}", "variants": variants}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )


def main() -> None:
    baseline = load("docs/experiments/32-spread-baselines.json")
    iql = variant("runs/marl-value/iql/history.json")
    shared = variant("runs/marl-value/shared-no-id/history.json")
    shared_id = variant("runs/marl-value/shared-id/history.json")
    vdn_shared = variant("runs/marl-value/vdn-shared/history.json")
    vdn_progress = variant("runs/marl-value/vdn-progress/history.json")
    qmix = variant("runs/marl-value/qmix/history.json")
    ippo = variant("runs/marl-ppo/ippo/history.json")
    mappo = variant("runs/marl-ppo/mappo/history.json")
    save(33, "independent-dqn", {"iql": iql, "random_baseline": baseline["summaries"]["random"]})
    save(
        34,
        "parameter-sharing",
        {"independent": iql, "shared": shared, "shared_agent_id": shared_id},
    )
    save(
        35,
        "vdn-credit-assignment",
        {"vdn_shared": vdn_shared, "vdn_assignment_progress": vdn_progress},
    )
    save(36, "qmix", {"iql": iql, "vdn": vdn_shared, "qmix": qmix})
    save(37, "ippo-vs-mappo", {"ippo": ippo, "mappo": mappo})
    print("compiled chapters 33-37")


if __name__ == "__main__":
    main()

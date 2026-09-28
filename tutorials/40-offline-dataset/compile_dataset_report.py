"""Combine the dataset manifest and behavior-policy training history for chapter 40."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    manifest = json.loads(Path("datasets/acrobot/manifest.json").read_text(encoding="utf-8"))
    behavior_history = json.loads(
        Path("runs/acrobot-online-dqn/seed-42/history.json").read_text(encoding="utf-8")
    )
    report = {
        "experiment": "40-offline-dataset",
        "manifest": manifest,
        "behavior_policy_history": behavior_history,
        "selected_checkpoints": {"medium": 50_000, "expert": 60_000},
    }
    path = Path("docs/experiments/40-offline-dataset.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["datasets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compare_ppo_sac import build_comparison


def load_report(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_comparison_uses_matching_experiments_and_expected_reliability() -> None:
    report = build_comparison(
        load_report("docs/experiments/10-continuous-ppo.json"),
        load_report("docs/experiments/11-sac.json"),
    )
    assert report["fairness"]["environment_steps_per_seed"] == 50_000
    assert report["reliable_training_seeds"] == {
        "threshold": 0.9,
        "ppo": 2,
        "sac": 3,
        "total": 3,
    }
    assert report["aggregate"]["sac"]["success_rate_mean"] == pytest.approx(0.9833333)


def test_comparison_rejects_different_evaluation_seeds() -> None:
    ppo = load_report("docs/experiments/10-continuous-ppo.json")
    sac = load_report("docs/experiments/11-sac.json")
    sac["experiment"]["evaluation_seeds"] = [999]
    with pytest.raises(ValueError, match="evaluation_seeds"):
        build_comparison(ppo, sac)

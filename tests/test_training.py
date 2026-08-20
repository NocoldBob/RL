from __future__ import annotations

from pathlib import Path

from main import TrainConfig, train


def test_short_training_run_creates_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    summary = train(
        TrainConfig(
            episodes=3,
            teacher_episodes=1,
            grid_size=5,
            end_score=3,
            max_steps=6,
            eval_interval=2,
            eval_episodes=2,
            save_interval=2,
            output_dir=output_dir,
            tensorboard=False,
        )
    )
    assert summary["episodes"] == 3
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert (output_dir / "summary.json").is_file()

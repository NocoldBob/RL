"""Generate lesson 20 partial-observation figures."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_cover(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.75), facecolor="#0F172A")
    axis.axis("off")
    axis.text(0.06, 0.78, "RL 强化学习从小白到老鸟（二十）", color="#93C5FD", fontsize=18)
    axis.text(0.06, 0.52, "视野缩小以后", color="white", fontsize=42, weight="bold")
    axis.text(0.06, 0.32, "同一画面可能需要不同动作", color="#FBBF24", fontsize=30, weight="bold")
    axis.text(0.80, 0.58, "7 × 7", color="#60A5FA", fontsize=52, weight="bold", ha="center")
    axis.text(0.80, 0.28, "局部视野", color="#C4B5FD", fontsize=22, ha="center")
    save(figure, path)


def plot_curves(report: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for mode, label, color in (("full", "完整地图", "#2563EB"), ("partial", "局部视野", "#F97316")):
        rows = report["histories"][mode]["evaluations"]
        axis.plot(
            [row["step"] for row in rows],
            [100 * row["success_rate"] for row in rows],
            marker="o",
            label=label,
            color=color,
        )
    axis.set_ylim(-2, 102)
    axis.set_xlabel("环境交互步数")
    axis.set_ylabel("成功率")
    axis.set_title("局部视野也能学会，但训练波动更剧烈", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, path)


def plot_best(report: dict[str, Any], path: Path) -> None:
    rows = report["variants"]
    labels = ["完整地图 best", "局部视野 best"]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        labels, [100 * row["success_rate"] for row in rows], color=["#2563EB", "#F97316"]
    )
    axis.bar_label(bars, fmt="%.0f%%", padding=3)
    axis.set_ylim(0, 100)
    axis.set_ylabel("100 个新起点成功率")
    axis.set_title("最佳检查点的泛化成绩", fontsize=16, weight="bold")
    axis.grid(axis="y", alpha=0.2)
    save(figure, path)


def plot_aliasing(report: dict[str, Any], path: Path) -> None:
    data = report["aliasing"]
    labels = ["唯一局部画面", "对应多个专家动作", "跨越持钥匙阶段"]
    values = [
        data["unique_local_observations"],
        data["observations_with_multiple_expert_actions"],
        data["observations_seen_before_and_after_key"],
    ]
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(labels, values, color=["#64748B", "#EF4444", "#8B5CF6"])
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("观测模式数量")
    axis.set_title("像素相同，不代表任务状态相同", fontsize=16, weight="bold")
    axis.grid(axis="y", alpha=0.2)
    save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_curves(report, output_dir / "learning-curves.png")
    plot_best(report, output_dir / "best-checkpoints.png")
    plot_aliasing(report, output_dir / "observation-aliasing.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lesson 20 figures")
    parser.add_argument(
        "--input", type=Path, default=Path("docs/experiments/20-doorkey-partial-dqn.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/csdn-20"))
    args = parser.parse_args()
    generate_assets(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()

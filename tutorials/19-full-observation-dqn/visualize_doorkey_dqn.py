"""Generate lesson 19 full-observation DQN figures."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_cover(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.75), facecolor="#111827")
    axis.axis("off")
    axis.text(0.06, 0.78, "RL 强化学习从小白到老鸟（十九）", color="#93C5FD", fontsize=18)
    axis.text(0.06, 0.52, "看见完整地图", color="white", fontsize=41, weight="bold")
    axis.text(0.06, 0.32, "DQN 就能稳定学会开门吗？", color="#FDBA74", fontsize=30, weight="bold")
    axis.text(0.82, 0.62, "50%", color="#60A5FA", fontsize=50, weight="bold", ha="center")
    axis.text(0.82, 0.38, "→ 4%", color="#FCA5A5", fontsize=40, weight="bold", ha="center")
    axis.text(0.82, 0.20, "学会后仍会退化", color="#C4B5FD", fontsize=16, ha="center")
    save(figure, path)


def plot_curves(report: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for mode, label, color in (
        ("sparse", "原始稀疏奖励", "#2563EB"),
        ("milestones", "里程碑奖励", "#F97316"),
    ):
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
    axis.set_ylabel("固定起点成功率")
    axis.set_title("稀疏奖励出现学习跃迁，也出现后期退化", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, path)


def plot_milestones(report: dict[str, Any], path: Path) -> None:
    rows = report["variants"]
    labels = ["稀疏奖励 best", "里程碑 best"]
    metrics = [
        ("key_pickup_rate", "拾取钥匙"),
        ("door_open_rate", "打开门"),
        ("success_rate", "到达终点"),
    ]
    x = np.arange(2)
    figure, axis = plt.subplots(figsize=(9, 5.2))
    width = 0.24
    for index, (key, label) in enumerate(metrics):
        bars = axis.bar(
            x + (index - 1) * width, [100 * row[key] for row in rows], width, label=label
        )
        axis.bar_label(bars, fmt="%.0f%%", padding=3)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 105)
    axis.set_ylabel("100 个新起点完成率")
    axis.set_title("奖励塑形改变了局部目标，却没有保证最终成功", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(figure, path)


def plot_diagnostics(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for mode, label, color in (("sparse", "稀疏", "#2563EB"), ("milestones", "里程碑", "#F97316")):
        rows = report["histories"][mode]["updates"]
        axes[0].plot(
            [row["step"] for row in rows],
            [row["loss"] for row in rows],
            alpha=0.7,
            label=label,
            color=color,
        )
        axes[1].plot(
            [row["step"] for row in rows],
            [row["q_mean"] for row in rows],
            alpha=0.7,
            label=label,
            color=color,
        )
    axes[0].set_title("Huber 损失")
    axes[1].set_title("已选动作平均 Q 值")
    for axis in axes:
        axis.set_xlabel("环境交互步数")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("损失下降并不能替代任务成功率", fontsize=16, weight="bold")
    save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_curves(report, output_dir / "learning-curves.png")
    plot_milestones(report, output_dir / "milestones.png")
    plot_diagnostics(report, output_dir / "diagnostics.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lesson 19 figures")
    parser.add_argument(
        "--input", type=Path, default=Path("docs/experiments/19-doorkey-full-dqn.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/csdn-19"))
    args = parser.parse_args()
    generate_assets(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()

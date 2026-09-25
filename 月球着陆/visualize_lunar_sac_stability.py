"""Generate lesson 16 SAC diagnostic figures."""

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
    figure, axis = plt.subplots(figsize=(12, 6.75), facecolor="#111827")
    axis.axis("off")
    axis.text(0.06, 0.78, "RL 强化学习从小白到老鸟（十六）", color="#93C5FD", fontsize=18)
    axis.text(0.06, 0.53, "网络更大、探索更多", color="white", fontsize=36, weight="bold")
    axis.text(0.06, 0.34, "SAC 就一定会着陆吗？", color="#FDBA74", fontsize=31, weight="bold")
    axis.text(0.78, 0.68, "Q1", color="#38BDF8", fontsize=42, weight="bold")
    axis.text(0.78, 0.40, "Q2", color="#A78BFA", fontsize=42, weight="bold")
    axis.text(0.78, 0.18, "0% 着陆", color="#FCA5A5", fontsize=20, weight="bold")
    save(figure, path)


def plot_learning(report: dict[str, Any], path: Path) -> None:
    rows = report["default_evaluations"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    steps = [row["step"] for row in rows]
    axes[0].plot(steps, [row["average_return"] for row in rows], marker="o", color="#2563EB")
    axes[0].set_title("平均回报逐步改善后回落")
    axes[1].plot(steps, [100 * row["landing_rate"] for row in rows], marker="o", color="#EF4444")
    axes[1].set_title("安全着陆率")
    axes[1].set_ylim(-2, 102)
    for axis in axes:
        axis.set_xlabel("环境交互步数")
        axis.grid(alpha=0.2)
    figure.suptitle("128×128 SAC：靠近目标不等于完成接地", fontsize=16, weight="bold")
    save(figure, path)


def plot_entropy_ablation(report: dict[str, Any], path: Path) -> None:
    rows = report["variants_at_50k"]
    labels = ["目标熵 -2", "目标熵 -1"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    bars = axes[0].bar(
        labels, [row["average_return"] for row in rows], color=["#2563EB", "#F97316"]
    )
    axes[0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0].set_title("5 万步平均回报")
    axes[1].bar(labels, [row["alpha"] for row in rows], color=["#2563EB", "#F97316"])
    axes[1].set_title("5 万步温度系数 α")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("提高目标熵保留了探索，却没有改善任务结果", fontsize=16, weight="bold")
    save(figure, path)


def plot_q_diagnostics(report: dict[str, Any], path: Path) -> None:
    rows = report["default_updates"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(
        [row["step"] for row in rows], [row["q_loss"] for row in rows], color="#0EA5E9", alpha=0.8
    )
    axes[0].set_title("双 Q 损失")
    axes[1].plot(
        [row["step"] for row in rows], [row["q_disagreement"] for row in rows], color="#8B5CF6"
    )
    axes[1].set_title("|Q1-Q2| 均值")
    for axis in axes:
        axis.set_xlabel("环境交互步数")
        axis.grid(alpha=0.2)
    figure.suptitle("Q 网络没有爆炸，但价值一致不代表策略会着陆", fontsize=16, weight="bold")
    save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_learning(report, output_dir / "learning-curves.png")
    plot_entropy_ablation(report, output_dir / "entropy-ablation.png")
    plot_q_diagnostics(report, output_dir / "q-diagnostics.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lesson 16 figures")
    parser.add_argument(
        "--input", type=Path, default=Path("docs/experiments/16-lunar-sac-stability.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/csdn-16"))
    args = parser.parse_args()
    generate_assets(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()

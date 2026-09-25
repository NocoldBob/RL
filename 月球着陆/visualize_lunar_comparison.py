"""Generate lesson 17 PPO/SAC comparison figures."""

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
    figure, axis = plt.subplots(figsize=(12, 6.75), facecolor="#0F172A")
    axis.axis("off")
    axis.text(0.06, 0.78, "RL 强化学习从小白到老鸟（十七）", color="#93C5FD", fontsize=18)
    axis.text(0.06, 0.52, "PPO 还是 SAC？", color="white", fontsize=42, weight="bold")
    axis.text(0.06, 0.32, "同预算与固定危险场景复盘", color="#FBBF24", fontsize=28, weight="bold")
    axis.text(0.78, 0.64, "PPO", color="#60A5FA", fontsize=36, weight="bold", ha="center")
    axis.text(0.78, 0.42, "VS", color="white", fontsize=24, weight="bold", ha="center")
    axis.text(0.78, 0.20, "SAC", color="#FB923C", fontsize=36, weight="bold", ha="center")
    save(figure, path)


def plot_common_budget(report: dict[str, Any], path: Path) -> None:
    results = report["common_budget"]["results"]
    labels = ["PPO", "SAC"]
    colors = ["#2563EB", "#F97316"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    bars = axes[0].bar(
        labels, [results[key]["average_return"] for key in ("ppo", "sac")], color=colors
    )
    axes[0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0].set_title("10 万步平均回报")
    bars = axes[1].bar(
        labels, [100 * results[key]["landing_rate"] for key in ("ppo", "sac")], color=colors
    )
    axes[1].bar_label(bars, fmt="%.0f%%", padding=3)
    axes[1].set_title("10 万步安全着陆率")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("相同 seed、步数和测试起点：PPO 领先", fontsize=16, weight="bold")
    save(figure, path)


def plot_deployment(report: dict[str, Any], path: Path) -> None:
    results = report["best_checkpoint_evaluation"]
    labels = ["PPO best", "SAC best"]
    colors = ["#2563EB", "#F97316"]
    summaries = [results["ppo"]["summary"], results["sac"]["summary"]]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].bar(labels, [item["average_return"] for item in summaries], color=colors)
    axes[0].set_title("100 个新起点平均回报")
    axes[1].bar(labels, [100 * item["landing_rate"] for item in summaries], color=colors)
    axes[1].set_title("100 个新起点着陆率")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("最佳检查点部署评估（训练预算不同）", fontsize=16, weight="bold")
    save(figure, path)


def plot_scenarios(report: dict[str, Any], path: Path) -> None:
    rows = report["fixed_scenarios"]
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    policies = ["heuristic", "ppo", "sac"]
    labels = ["规则", "PPO", "SAC"]
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(scenarios))
    width = 0.24
    colors = ["#16A34A", "#2563EB", "#F97316"]
    for index, (policy, label, color) in enumerate(zip(policies, labels, colors, strict=True)):
        selected = [
            next(row for row in rows if row["scenario"] == scene and row["policy"] == policy)
            for scene in scenarios
        ]
        axes[0].bar(
            x + (index - 1) * width,
            [row["main_action"] for row in selected],
            width,
            label=label,
            color=color,
        )
        axes[1].bar(
            x + (index - 1) * width,
            [row["side_action"] for row in selected],
            width,
            label=label,
            color=color,
        )
    axes[0].set_ylabel("主发动机动作")
    axes[1].set_ylabel("侧向发动机动作")
    axes[1].set_xticks(x, ["左侧外飘", "右侧倾斜", "近地快降", "单腿接触", "双腿接触"])
    for axis in axes:
        axis.axhline(0, color="#334155", linewidth=1)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(ncol=3)
    figure.suptitle("同一危险状态，三种控制器会怎么点火", fontsize=16, weight="bold")
    save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_common_budget(report, output_dir / "common-budget.png")
    plot_deployment(report, output_dir / "deployment.png")
    plot_scenarios(report, output_dir / "fixed-scenarios.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lesson 17 figures")
    parser.add_argument(
        "--input", type=Path, default=Path("docs/experiments/17-lunar-ppo-vs-sac.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/csdn-17"))
    args = parser.parse_args()
    generate_assets(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()

"""Generate article figures for lesson 11 SAC."""

# ruff: noqa: RUF001

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

COLORS = {"blue": "#2563EB", "orange": "#F97316", "green": "#16A34A", "gray": "#475569"}


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_architecture(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7)
    axis.axis("off")
    boxes = [
        (0.5, 2.7, 2.0, 1.4, "Replay Buffer\n经验反复使用", COLORS["gray"]),
        (3.3, 4.7, 2.0, 1.2, "Actor\n连续动作分布", COLORS["blue"]),
        (3.3, 1.1, 2.0, 1.2, "自动温度 α\n调节随机性", COLORS["orange"]),
        (6.2, 4.7, 2.0, 1.2, "Q1", COLORS["green"]),
        (6.2, 2.8, 2.0, 1.2, "Q2", COLORS["green"]),
        (9.1, 3.7, 2.2, 1.4, "Target Q1/Q2\n软更新", COLORS["gray"]),
    ]
    for x, y, width, height, label, color in boxes:
        axis.add_patch(plt.Rectangle((x, y), width, height, color=color, alpha=0.92, linewidth=0))
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=12,
        )
    arrows = [
        ((2.5, 3.4), (3.3, 5.3)),
        ((2.5, 3.4), (6.2, 5.3)),
        ((2.5, 3.4), (6.2, 3.4)),
        ((5.3, 5.3), (6.2, 5.3)),
        ((5.3, 1.7), (4.3, 4.7)),
        ((8.2, 4.4), (9.1, 4.4)),
    ]
    for start, end in arrows:
        axis.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#64748B"}
        )
    axis.set_title("SAC：回放、双 Q 网络与熵共同组成的离策略学习", fontsize=18, weight="bold")
    save_figure(figure, path)


def plot_exploration(report: dict[str, Any], path: Path) -> None:
    rows = report["exploration_probe"]
    repeats = [str(row["action_repeat"]) for row in rows]
    success = [100 * row["success_rate"] for row in rows]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(
        repeats, success, color=[COLORS["gray"], COLORS["blue"], COLORS["green"], COLORS["orange"]]
    )
    axis.bar_label(bars, fmt="%.0f%%", padding=4)
    axis.set_ylim(0, 105)
    axis.set_xlabel("随机动作保持步数")
    axis.set_ylabel("发现终点的回合比例")
    axis.set_title("惯性任务需要时间相关探索", fontsize=16, weight="bold")
    axis.grid(axis="y", alpha=0.2)
    save_figure(figure, path)


def plot_training(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for seed, history in report["sac"]["histories"].items():
        evaluations = history["evaluations"]
        steps = [row["step"] for row in evaluations]
        axes[0].plot(
            steps, [row["average_return"] for row in evaluations], marker="o", label=f"seed={seed}"
        )
        axes[1].plot(
            steps,
            [100 * row["success_rate"] for row in evaluations],
            marker="o",
            label=f"seed={seed}",
        )
    axes[0].set_title("确定性评估回报")
    axes[0].set_ylabel("平均回报")
    axes[1].set_title("确定性评估成功率")
    axes[1].set_ylabel("成功率（%）")
    for axis in axes:
        axis.set_xlabel("环境交互步数")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("SAC 多随机种子训练过程", fontsize=17, weight="bold")
    save_figure(figure, path)


def plot_seed_results(report: dict[str, Any], path: Path) -> None:
    seeds = report["sac"]["training_seeds"]
    labels = [str(row["training_seed"]) for row in seeds]
    returns = [row["summary"]["average_return"] for row in seeds]
    successes = [100 * row["summary"]["success_rate"] for row in seeds]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].bar(x, returns, color=COLORS["blue"])
    axes[1].bar(x, successes, color=COLORS["green"])
    axes[0].set_title("平均回报")
    axes[1].set_title("成功率")
    axes[1].set_ylim(0, 105)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("训练种子")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("同一套 SAC 配置，不同训练种子的最终结果", fontsize=16, weight="bold")
    save_figure(figure, path)


def plot_trajectories(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(
        len(report["sac"]["training_seeds"]), 1, figsize=(11, 8), sharex=False
    )
    for axis, row in zip(axes, report["sac"]["training_seeds"], strict=True):
        trace = row["trace"]
        steps = np.arange(len(trace["action"]))
        axis.plot(
            np.arange(len(trace["position"])), trace["position"], label="位置", color=COLORS["blue"]
        )
        axis.plot(steps, trace["action"], label="油门", color=COLORS["orange"], alpha=0.75)
        axis.axhline(0.45, color=COLORS["green"], linestyle="--", alpha=0.6)
        axis.set_title(f"seed={row['training_seed']}")
        axis.grid(alpha=0.18)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("时间步")
    figure.suptitle("相同初始状态下的 SAC 轨迹", fontsize=17, weight="bold")
    save_figure(figure, path)


def plot_cover(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.75))
    axis.set_facecolor("#0F172A")
    figure.patch.set_facecolor("#0F172A")
    axis.axis("off")
    axis.text(
        0.06,
        0.72,
        "RL 强化学习从小白到老鸟（十一）",
        transform=axis.transAxes,
        color="#93C5FD",
        fontsize=18,
    )
    axis.text(
        0.06,
        0.48,
        "经验也能反复学习",
        transform=axis.transAxes,
        color="white",
        fontsize=35,
        weight="bold",
    )
    axis.text(
        0.06,
        0.32,
        "第一次实现 SAC",
        transform=axis.transAxes,
        color="#FDBA74",
        fontsize=28,
        weight="bold",
    )
    axis.text(
        0.67,
        0.53,
        "Actor",
        transform=axis.transAxes,
        color="white",
        fontsize=20,
        ha="center",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": COLORS["blue"], "edgecolor": "none"},
    )
    axis.text(
        0.84,
        0.66,
        "Q1",
        transform=axis.transAxes,
        color="white",
        fontsize=18,
        ha="center",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": COLORS["green"], "edgecolor": "none"},
    )
    axis.text(
        0.84,
        0.40,
        "Q2",
        transform=axis.transAxes,
        color="white",
        fontsize=18,
        ha="center",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": COLORS["green"], "edgecolor": "none"},
    )
    axis.annotate(
        "",
        xy=(0.78, 0.63),
        xytext=(0.72, 0.56),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#CBD5E1", "lw": 2},
    )
    axis.annotate(
        "",
        xy=(0.78, 0.43),
        xytext=(0.72, 0.51),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#CBD5E1", "lw": 2},
    )
    save_figure(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_architecture(output_dir / "sac-architecture.png")
    plot_exploration(report, output_dir / "exploration-repeat.png")
    plot_training(report, output_dir / "training-curves.png")
    plot_seed_results(report, output_dir / "seed-results.png")
    plot_trajectories(report, output_dir / "trajectories.png")

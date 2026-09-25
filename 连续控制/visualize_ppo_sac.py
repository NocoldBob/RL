"""Generate lesson 12 PPO versus SAC comparison figures."""

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

PPO_COLOR = "#2563EB"
SAC_COLOR = "#F97316"


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_cover(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.75), facecolor="#F8FAFC")
    axis.axis("off")
    axis.text(
        0.06,
        0.76,
        "RL 强化学习从小白到老鸟（十二）",
        transform=axis.transAxes,
        fontsize=18,
        color="#64748B",
    )
    axis.text(
        0.06,
        0.55,
        "PPO 还是 SAC？",
        transform=axis.transAxes,
        fontsize=38,
        weight="bold",
        color="#0F172A",
    )
    axis.text(
        0.06,
        0.38,
        "同样 5 万步，结果为什么不同",
        transform=axis.transAxes,
        fontsize=25,
        color="#475569",
    )
    axis.text(
        0.71,
        0.62,
        "PPO",
        transform=axis.transAxes,
        fontsize=30,
        weight="bold",
        color="white",
        ha="center",
        bbox={"boxstyle": "round,pad=0.7", "facecolor": PPO_COLOR, "edgecolor": "none"},
    )
    axis.text(
        0.71,
        0.30,
        "SAC",
        transform=axis.transAxes,
        fontsize=30,
        weight="bold",
        color="white",
        ha="center",
        bbox={"boxstyle": "round,pad=0.7", "facecolor": SAC_COLOR, "edgecolor": "none"},
    )
    axis.text(
        0.87,
        0.46,
        "50k\nsteps",
        transform=axis.transAxes,
        fontsize=20,
        weight="bold",
        color="#0F172A",
        ha="center",
        va="center",
    )
    save(figure, path)


def plot_success_by_seed(report: dict[str, Any], path: Path) -> None:
    rows = report["paired_training_seeds"]
    labels = [str(row["training_seed"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    figure, axis = plt.subplots(figsize=(9, 5.5))
    ppo = axis.bar(
        x - width / 2,
        [100 * row["ppo_success_rate"] for row in rows],
        width,
        label="PPO",
        color=PPO_COLOR,
    )
    sac = axis.bar(
        x + width / 2,
        [100 * row["sac_success_rate"] for row in rows],
        width,
        label="SAC",
        color=SAC_COLOR,
    )
    axis.bar_label(ppo, fmt="%.0f%%", padding=3)
    axis.bar_label(sac, fmt="%.0f%%", padding=3)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 108)
    axis.set_xlabel("训练种子")
    axis.set_ylabel("100 个测试起点的成功率")
    axis.set_title("SAC 消除了本次实验中的整种子失败", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(figure, path)


def plot_learning_curves(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for axis, seed in zip(axes, (7, 42, 2026), strict=True):
        for algorithm, color in (("ppo", PPO_COLOR), ("sac", SAC_COLOR)):
            evaluations = report["histories"][algorithm][str(seed)]["evaluations"]
            axis.plot(
                [row["step"] for row in evaluations],
                [100 * row["success_rate"] for row in evaluations],
                marker="o",
                color=color,
                label=algorithm.upper(),
            )
        axis.set_title(f"seed={seed}")
        axis.set_xlabel("环境交互步数")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("阶段评估成功率（%）")
    axes[0].legend()
    figure.suptitle("更早学会与最终更可靠，不是同一件事", fontsize=17, weight="bold")
    save(figure, path)


def plot_tradeoffs(report: dict[str, Any], path: Path) -> None:
    rows = report["paired_training_seeds"]
    figure, axis = plt.subplots(figsize=(9, 6))
    for algorithm, color in (("ppo", PPO_COLOR), ("sac", SAC_COLOR)):
        for row in rows:
            steps = row[f"{algorithm}_average_success_steps"]
            if steps is None:
                continue
            cost = row[f"{algorithm}_average_action_cost"]
            axis.scatter(steps, cost, s=130, color=color)
            axis.annotate(
                f"{algorithm.upper()}-{row['training_seed']}",
                (steps, cost),
                xytext=(6, 6),
                textcoords="offset points",
            )
    axis.set_xlabel("成功回合平均步数（越小越快）")
    axis.set_ylabel("平均动作成本（越小越省）")
    axis.set_title("SAC 更快，PPO 更省油", fontsize=16, weight="bold")
    axis.grid(alpha=0.2)
    save(figure, path)


def plot_aggregate(report: dict[str, Any], path: Path) -> None:
    ppo = report["aggregate"]["ppo"]
    sac = report["aggregate"]["sac"]
    labels = ["成功率", "平均回报", "可靠种子"]
    ppo_values = [ppo["success_rate_mean"] * 100, ppo["average_return_mean"], 2 / 3 * 100]
    sac_values = [sac["success_rate_mean"] * 100, sac["average_return_mean"], 3 / 3 * 100]
    x = np.arange(3)
    width = 0.34
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(x - width / 2, ppo_values, width, color=PPO_COLOR, label="PPO")
    axis.bar(x + width / 2, sac_values, width, color=SAC_COLOR, label="SAC")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 110)
    axis.set_ylabel("归一到百分制的实验指标")
    axis.set_title("5 万环境步后的整体结果", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_success_by_seed(report, output_dir / "success-by-seed.png")
    plot_learning_curves(report, output_dir / "learning-curves.png")
    plot_tradeoffs(report, output_dir / "speed-cost-tradeoff.png")
    plot_aggregate(report, output_dir / "aggregate-results.png")

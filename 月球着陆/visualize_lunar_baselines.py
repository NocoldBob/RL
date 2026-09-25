"""Generate lesson 13 LunarLander baseline figures."""

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

COLORS = {
    "zero": "#94A3B8",
    "random": "#A855F7",
    "heuristic": "#16A34A",
    "noisy_heuristic": "#F97316",
}
LABELS = {
    "zero": "零推力",
    "random": "随机控制",
    "heuristic": "规则控制",
    "noisy_heuristic": "带噪规则",
}


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_cover(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.75), facecolor="#07111F")
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7)
    axis.axis("off")
    axis.text(0.7, 5.7, "RL 强化学习从小白到老鸟（十三）", color="#93C5FD", fontsize=18)
    axis.text(0.7, 4.1, "一根油门不够了", color="white", fontsize=38, weight="bold")
    axis.text(0.7, 2.9, "走进二维连续动作", color="#FDBA74", fontsize=28, weight="bold")
    axis.plot([8.3, 10.5, 9.4, 8.3], [2.2, 2.2, 4.8, 2.2], color="#E2E8F0", linewidth=4)
    axis.plot([8.7, 8.0], [2.2, 1.3], color="#E2E8F0", linewidth=3)
    axis.plot([10.1, 10.8], [2.2, 1.3], color="#E2E8F0", linewidth=3)
    axis.arrow(9.4, 2.2, 0, -1.0, width=0.08, color="#F97316", length_includes_head=True)
    axis.arrow(8.9, 3.2, -0.9, 0, width=0.06, color="#38BDF8", length_includes_head=True)
    axis.text(9.4, 0.5, "主发动机 + 侧向发动机", color="#CBD5E1", fontsize=15, ha="center")
    save(figure, path)


def plot_state_action(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7)
    axis.axis("off")
    state_labels = [
        "水平位置 x",
        "高度 y",
        "水平速度 vx",
        "垂直速度 vy",
        "姿态角",
        "角速度",
        "左腿接触",
        "右腿接触",
    ]
    for index, label in enumerate(state_labels):
        row, column = divmod(index, 4)
        x = 0.4 + column * 2.25
        y = 5.0 - row * 1.25
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#475569", "edgecolor": "none"},
        )
    axis.annotate(
        "",
        xy=(9.5, 4.3),
        xytext=(8.3, 4.3),
        arrowprops={"arrowstyle": "->", "lw": 2, "color": "#64748B"},
    )
    axis.text(
        10.6,
        4.9,
        "动作 0\n主发动机",
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F97316", "edgecolor": "none"},
    )
    axis.text(
        10.6,
        3.1,
        "动作 1\n侧向发动机",
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#2563EB", "edgecolor": "none"},
    )
    axis.text(
        0.4,
        1.1,
        "动作范围都是 [-1, 1]，但两个维度的死区和动力含义并不相同",
        fontsize=15,
        weight="bold",
    )
    axis.set_title("8 维状态 → 2 维连续控制", fontsize=18, weight="bold")
    save(figure, path)


def plot_summary(report: dict[str, Any], path: Path) -> None:
    policies = list(report["aggregate"])
    x = np.arange(len(policies))
    colors = [COLORS[name] for name in policies]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    landing = [100 * report["aggregate"][name]["landing_rate"] for name in policies]
    returns = [report["aggregate"][name]["average_return"] for name in policies]
    bars = axes[0].bar(x, landing, color=colors)
    axes[0].bar_label(bars, fmt="%.0f%%", padding=3)
    axes[0].set_ylim(0, 108)
    axes[0].set_title("安全着陆率")
    axes[1].bar(x, returns, color=colors)
    axes[1].axhline(200, linestyle="--", color="#16A34A", label="环境解决线 200")
    axes[1].set_title("平均回报")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, [LABELS[name] for name in policies], rotation=12)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("四种二维控制基线", fontsize=17, weight="bold")
    save(figure, path)


def plot_trajectories(report: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    for policy, trace in report["traces"].items():
        axis.plot(trace["x"], trace["y"], label=LABELS[policy], color=COLORS[policy], linewidth=2)
        axis.scatter(trace["x"][0], trace["y"][0], color=COLORS[policy], s=35)
    axis.axhline(0, color="#334155", linewidth=2)
    axis.axvspan(-0.2, 0.2, color="#86EFAC", alpha=0.2, label="着陆区附近")
    axis.set_xlabel("归一化水平位置")
    axis.set_ylabel("归一化高度")
    axis.set_title("相同初始状态下的飞行轨迹", fontsize=16, weight="bold")
    axis.grid(alpha=0.2)
    axis.legend()
    save(figure, path)


def plot_controls(report: dict[str, Any], path: Path) -> None:
    trace = report["traces"]["heuristic"]
    steps = np.arange(len(trace["main_action"]))
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(steps, trace["main_action"], color="#F97316")
    axes[0].axhline(0, color="#64748B", linewidth=1)
    axes[0].set_ylabel("主发动机动作")
    axes[1].plot(steps, trace["side_action"], color="#2563EB")
    axes[1].axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1)
    axes[1].axhline(-0.5, color="#94A3B8", linestyle="--", linewidth=1)
    axes[1].set_ylabel("侧向发动机动作")
    axes[1].set_xlabel("时间步")
    for axis in axes:
        axis.set_ylim(-1.08, 1.08)
        axis.grid(alpha=0.2)
    figure.suptitle("规则控制器如何同时管理下降与姿态", fontsize=17, weight="bold")
    save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_state_action(output_dir / "state-action.png")
    plot_summary(report, output_dir / "baseline-results.png")
    plot_trajectories(report, output_dir / "trajectories.png")
    plot_controls(report, output_dir / "heuristic-controls.png")

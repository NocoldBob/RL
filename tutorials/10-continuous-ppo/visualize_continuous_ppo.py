"""Generate the charts used by continuous PPO lesson 10."""

# ruff: noqa: RUF001

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

COLORS = {
    "ink": "#17212b",
    "green": "#23a36d",
    "orange": "#ef8354",
    "blue": "#3d7ea6",
    "yellow": "#f4c95d",
    "gray": "#84919d",
    "paper": "#f7f9f8",
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _moving_average(values: list[float], window: int = 10) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < window:
        return array
    return np.convolve(array, np.ones(window) / window, mode="valid")


def draw_cover(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.3), facecolor=COLORS["paper"])
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-0.72, 0.78)
    axis.axis("off")
    x = np.linspace(-1.2, 1.2, 500)
    y = 0.42 * x**2 - 0.48
    axis.plot(x, y, color=COLORS["ink"], linewidth=5)
    car_x = 0.32
    car_y = 0.42 * car_x**2 - 0.43
    axis.scatter(
        [car_x], [car_y], s=950, color=COLORS["green"], edgecolor="white", linewidth=4, zorder=5
    )
    for offset, width, color in [
        (-0.33, 0.22, COLORS["orange"]),
        (-0.53, 0.14, COLORS["yellow"]),
        (-0.68, 0.08, COLORS["blue"]),
    ]:
        axis.plot(
            [car_x + offset, car_x + offset + width],
            [car_y, car_y],
            color=color,
            linewidth=12,
            solid_capstyle="round",
        )
    axis.text(-1.15, 0.59, "CONTINUOUS PPO", fontsize=19, weight="bold", color=COLORS["green"])
    axis.text(-1.15, 0.34, "油门到底该踩多少？", fontsize=31, weight="bold", color=COLORS["ink"])
    axis.text(-1.15, 0.13, "从高斯分布到可控的连续动作", fontsize=17, color=COLORS["gray"])
    _save(figure, path)


def draw_policy_pipeline(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 4.3), facecolor="white")
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4.3)
    axis.axis("off")
    boxes = [
        (0.35, "位置、速度\n2 个状态", COLORS["blue"]),
        (3.2, "策略网络\n64 × 64", COLORS["green"]),
        (6.05, "高斯分布\nμ 与 σ", COLORS["orange"]),
        (8.9, "tanh 压缩\n[-1, 1] 油门", COLORS["yellow"]),
    ]
    for x, text, color in boxes:
        patch = FancyBboxPatch(
            (x, 1.35),
            2.2,
            1.45,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="none",
            alpha=0.92,
        )
        axis.add_patch(patch)
        axis.text(
            x + 1.1,
            2.08,
            text,
            ha="center",
            va="center",
            fontsize=15,
            weight="bold",
            color="white" if color != COLORS["yellow"] else COLORS["ink"],
        )
    for x in (2.58, 5.43, 8.28):
        axis.add_patch(
            FancyArrowPatch(
                (x, 2.08),
                (x + 0.52, 2.08),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2,
                color=COLORS["ink"],
            )
        )
    axis.text(
        6,
        3.55,
        "连续 PPO 的动作生成过程",
        ha="center",
        fontsize=22,
        weight="bold",
        color=COLORS["ink"],
    )
    axis.text(
        6,
        0.55,
        "训练时采样保持探索，评估时使用 tanh(μ) 得到稳定动作",
        ha="center",
        fontsize=13,
        color=COLORS["gray"],
    )
    _save(figure, path)


def draw_training_curves(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")
    for seed, history in report["ppo"]["histories"].items():
        episodes = history["episodes"]
        returns = [row["return"] for row in episodes]
        smoothed = _moving_average(returns, 10)
        steps = [row["step"] for row in episodes][-len(smoothed) :]
        axes[0].plot(steps, smoothed, label=f"seed={seed}", linewidth=1.8)
        evaluations = history["evaluations"]
        axes[1].plot(
            [row["step"] for row in evaluations],
            [100 * row["success_rate"] for row in evaluations],
            marker="o",
            label=f"seed={seed}",
        )
    axes[0].set(title="训练回报（10 回合移动平均）", xlabel="环境交互步数", ylabel="回报")
    axes[1].set(
        title="固定评估集成功率", xlabel="环境交互步数", ylabel="成功率 (%)", ylim=(-3, 103)
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, path)


def draw_comparison(report: dict[str, Any], path: Path) -> None:
    baselines = report["baselines"]
    ppo = report["ppo"]["aggregate_across_training_seeds"]
    names = ["零油门", "随机油门", "全油门惯性", "平滑惯性", "连续 PPO"]
    keys = ["zero", "random", "bang_bang", "smooth_momentum"]
    returns = [baselines[key]["average_return"] for key in keys] + [ppo["average_return_mean"]]
    success = [100 * baselines[key]["success_rate"] for key in keys] + [
        100 * ppo["success_rate_mean"]
    ]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["yellow"], COLORS["green"]]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")
    axes[0].bar(names, returns, color=colors)
    axes[1].bar(names, success, color=colors)
    axes[0].set_title("平均回报")
    axes[1].set_title("成功率 (%)")
    axes[1].set_ylim(0, 108)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    _save(figure, path)


def draw_trajectories(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, facecolor="white")
    for item in report["ppo"]["training_seeds"]:
        trace = item["trace"]
        seed = item["training_seed"]
        axes[0].plot(trace["position"], label=f"训练 seed={seed}", linewidth=1.8)
        axes[1].plot(trace["action"], label=f"训练 seed={seed}", linewidth=1.5, alpha=0.9)
    axes[0].axhline(0.45, color=COLORS["orange"], linestyle="--", label="目标位置")
    axes[0].set_ylabel("位置")
    axes[1].set_ylabel("油门")
    axes[1].set_xlabel("步骤")
    axes[1].set_ylim(-1.08, 1.08)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=4)
    figure.suptitle("同一起点 seed=42 的确定性策略轨迹", fontsize=18, weight="bold")
    figure.tight_layout()
    _save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_cover(output_dir / "cover.png")
    draw_policy_pipeline(output_dir / "policy-pipeline.png")
    draw_training_curves(report, output_dir / "training-curves.png")
    draw_comparison(report, output_dir / "baseline-comparison.png")
    draw_trajectories(report, output_dir / "ppo-trajectories.png")

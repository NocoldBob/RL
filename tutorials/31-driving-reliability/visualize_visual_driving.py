"""Generate publication figures for visual-driving lessons 25-31."""

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

TITLES = {
    25: "从状态向量到 RGB 图像",
    26: "一张图够不够判断车辆状态",
    27: "离散化方向盘后 DQN 能学会开车吗",
    28: "连续动作 PPO 如何同时控制三件事",
    29: "奖励越精细，驾驶就一定越好吗",
    30: "画面变暗以后策略还认识赛道吗",
    31: "训练成功不等于可靠驾驶",
}


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def cover(lesson: int, subtitle: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.75), facecolor="#0B1220")
    axis.axis("off")
    axis.text(0.06, 0.80, f"RL 强化学习从小白到老鸟（{lesson}）", color="#7DD3FC", fontsize=18)
    axis.text(0.06, 0.53, TITLES[lesson], color="white", fontsize=34, weight="bold")
    axis.text(0.06, 0.33, subtitle, color="#FBBF24", fontsize=23, weight="bold")
    theta = np.linspace(0, 2 * np.pi, 200)
    axis.plot(0.82 + 0.10 * np.cos(theta), 0.52 + 0.28 * np.sin(theta), color="#334155", lw=16)
    axis.scatter([0.82], [0.76], s=500, marker="s", color="#38BDF8")
    save(figure, path)


def bar_chart(labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(
        labels, values, color=["#64748B", "#0EA5E9", "#F97316", "#16A34A"][: len(labels)]
    )
    axis.bar_label(bars, fmt="%.1f", padding=3)
    axis.set_title(title, fontsize=16, weight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.2)
    save(figure, path)


def learning_curves(variants: dict[str, Any], labels: dict[str, str], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for name, payload in variants.items():
        history = payload.get("history", payload)["evaluations"]
        axis.plot(
            [row["step"] for row in history],
            [row["average_return"] for row in history],
            marker="o",
            label=labels.get(name, name),
        )
    axis.axhline(200, color="#16A34A", ls="--", label="成功阈值 200")
    axis.set_xlabel("决策步数")
    axis.set_ylabel("固定赛道平均回报")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, path)


def lesson25(report: dict[str, Any], output: Path) -> None:
    cover(25, "96×96 RGB → 64×64 灰度帧堆叠", output / "cover.png")
    summaries = report["summaries"]
    names = list(summaries)
    labels = ["静止", "直行", "随机", "轨迹跟随"]
    bar_chart(
        labels,
        [summaries[name]["average_return"] for name in names],
        "动作存在，不等于驾驶成立",
        "平均回报",
        output / "baseline-returns.png",
    )
    bar_chart(
        labels,
        [100 * summaries[name]["offroad_rate"] for name in names],
        "近车区域的越界比例",
        "越界帧比例（%）",
        output / "offroad-rates.png",
    )
    cover(25, "视觉输入管线是算法之前的第一层实验", output / "conclusion.png")


def lesson26(report: dict[str, Any], output: Path) -> None:
    cover(26, "单帧看位置，连续帧看运动", output / "cover.png")
    learning_curves(
        report["variants"],
        {"single_frame": "单帧", "four_frames": "4 帧堆叠"},
        output / "learning-curves.png",
    )
    bar_chart(
        ["单帧", "4 帧"],
        [report["variants"][name]["best"]["average_return"] for name in report["variants"]],
        "相同预算下的最佳平均回报",
        "平均回报",
        output / "best-returns.png",
    )
    cover(26, "帧堆叠提供速度线索，也增加了输入和训练成本", output / "conclusion.png")


def lesson27(report: dict[str, Any], output: Path) -> None:
    cover(27, "9 个组合动作换来可复用经验", output / "cover.png")
    learning_curves(
        {"dqn": report["history"]}, {"dqn": "Double DQN"}, output / "learning-curves.png"
    )
    baselines = report["baseline_summaries"]
    bar_chart(
        ["随机", "轨迹跟随", "DQN"],
        [
            baselines["random"]["average_return"],
            baselines["privileged_local_follower"]["average_return"],
            report["best"]["average_return"],
        ],
        "学习策略仍需同时对照下限与特权参考",
        "平均回报",
        output / "baseline-comparison.png",
    )
    cover(27, "动作离散化降低难度，也限制控制精度", output / "conclusion.png")


def lesson28(report: dict[str, Any], output: Path) -> None:
    cover(28, "转向、油门、刹车联合采样", output / "cover.png")
    learning_curves(
        report["variants"],
        {"discrete_double_dqn": "离散 Double DQN", "continuous_ppo": "连续 PPO"},
        output / "learning-curves.png",
    )
    bar_chart(
        ["Double DQN", "连续 PPO"],
        [row["best"]["average_return"] for row in report["variants"].values()],
        "相同视觉管线，不同动作建模",
        "最佳平均回报",
        output / "best-returns.png",
    )
    cover(28, "连续控制消除了动作格点，但优化问题更难", output / "conclusion.png")


def lesson29(report: dict[str, Any], output: Path) -> None:
    cover(29, "塑形奖励会改变最优行为", output / "cover.png")
    learning_curves(
        report["variants"],
        {"raw_reward": "原始奖励", "shaped_reward": "塑形奖励"},
        output / "learning-curves.png",
    )
    bar_chart(
        ["原始奖励", "塑形奖励"],
        [row["best"]["average_return"] for row in report["variants"].values()],
        "最终仍用环境原始回报评价",
        "最佳原始回报",
        output / "best-returns.png",
    )
    cover(29, "训练信号可以变，评价口径不能跟着变", output / "conclusion.png")


def lesson30(report: dict[str, Any], output: Path) -> None:
    cover(30, "亮度、噪声、颜色、模糊与遮挡", output / "cover.png")
    agents = list(report["results"])
    corruptions = list(report["results"][agents[0]])
    matrix = np.asarray(
        [
            [report["results"][agent][kind]["summary"]["average_return"] for kind in corruptions]
            for agent in agents
        ]
    )
    figure, axis = plt.subplots(figsize=(10, 4.2))
    image = axis.imshow(matrix, cmap="RdYlGn", aspect="auto")
    axis.set_xticks(range(len(corruptions)), ["干净", "变暗", "噪声", "偏色", "模糊", "遮挡"])
    axis.set_yticks(range(len(agents)), ["普通 PPO", "增强 PPO"])
    for row in range(len(agents)):
        for column in range(len(corruptions)):
            axis.text(column, row, f"{matrix[row, column]:.0f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="平均回报")
    axis.set_title("图像扰动鲁棒性矩阵", fontsize=16, weight="bold")
    save(figure, output / "robustness-heatmap.png")
    clean = matrix[:, 0]
    degraded = matrix[:, 1:].mean(axis=1)
    bar_chart(
        ["普通 PPO", "增强 PPO"],
        (clean - degraded).tolist(),
        "从干净画面到扰动画面的平均性能下降",
        "回报下降",
        output / "average-degradation.png",
    )
    cover(30, "数据增强保护的是不变性，不保证驾驶能力", output / "conclusion.png")


def lesson31(report: dict[str, Any], output: Path) -> None:
    cover(31, "平均回报之外，还要看最差轨迹", output / "cover.png")
    results = report["results"]
    labels = ["DQN", "PPO", "塑形 PPO", "增强 PPO"]
    bar_chart(
        labels,
        [row["summary"]["average_return"] for row in results.values()],
        "未见赛道上的平均回报",
        "平均回报",
        output / "average-returns.png",
    )
    figure, axis = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(labels))
    axis.bar(x - 0.2, [row["median_return"] for row in results.values()], 0.4, label="中位数")
    axis.bar(x + 0.2, [row["p10_return"] for row in results.values()], 0.4, label="10% 分位")
    axis.set_xticks(x, labels)
    axis.set_ylabel("回报")
    axis.set_title("尾部表现揭示平均数隐藏的风险", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(figure, output / "tail-risk.png")
    cover(31, "可靠性是分布，不是一次最高分", output / "conclusion.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lesson", type=int, choices=range(25, 32), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    globals()[f"lesson{args.lesson}"](report, args.output_dir)


if __name__ == "__main__":
    main()

"""Generate figures for lessons 21-23."""

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


def cover(lesson: int, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.75), facecolor="#0F172A")
    axis.axis("off")
    axis.text(0.06, 0.78, f"RL 强化学习从小白到老鸟（{lesson}）", color="#93C5FD", fontsize=18)
    axis.text(0.06, 0.50, title, color="white", fontsize=37, weight="bold")
    axis.text(0.06, 0.29, "MiniGrid DoorKey", color="#FBBF24", fontsize=28, weight="bold")
    axis.plot([0.72, 0.90], [0.68, 0.33], color="#60A5FA", linewidth=8)
    axis.scatter([0.72, 0.90], [0.68, 0.33], color="#FBBF24", s=300)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    save(figure, path)


def lesson21(report: dict[str, Any], output: Path) -> None:
    cover(21, "记忆不是自动学会的", output / "cover.png")
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for name, label, color in (
        ("feedforward", "前馈 PPO", "#2563EB"),
        ("gru-truncated-bptt", "GRU PPO", "#F97316"),
    ):
        rows = report["histories"][name]["evaluations"]
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
    axis.set_title("前馈策略学会了，GRU 尚未学会如何使用记忆", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, output / "learning-curves.png")
    labels = ["前馈 PPO", "GRU 单步", "GRU 截断 BPTT"]
    values = [
        100 * report["variants"][key]["best_success_rate"]
        for key in ("feedforward", "gru-naive-single-step", "gru-truncated-bptt")
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=["#2563EB", "#94A3B8", "#F97316"])
    axis.bar_label(bars, fmt="%.0f%%", padding=3)
    axis.set_ylim(0, 105)
    axis.set_ylabel("最佳固定起点成功率")
    axis.grid(axis="y", alpha=0.2)
    save(figure, output / "best-results.png")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for name, label, color in (
        ("feedforward", "前馈", "#2563EB"),
        ("gru-truncated-bptt", "GRU", "#F97316"),
    ):
        rows = report["histories"][name]["updates"]
        axes[0].plot(
            [row["step"] for row in rows],
            [row["entropy"] for row in rows],
            label=label,
            color=color,
        )
        axes[1].plot(
            [row["step"] for row in rows],
            [row["approx_kl"] for row in rows],
            label=label,
            color=color,
        )
    axes[0].set_title("策略熵")
    axes[1].set_title("近似 KL")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    save(figure, output / "update-diagnostics.png")
    cover(21, "记住什么比拥有记忆更难", output / "conclusion.png")


def lesson22(report: dict[str, Any], output: Path) -> None:
    cover(22, "探索范围不等于任务进展", output / "cover.png")
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for name, label, color in (
        ("direct-sparse-8x8", "直接稀疏", "#64748B"),
        ("count-bonus-8x8", "访问计数", "#F97316"),
        ("curriculum-8x8", "课程学习", "#16A34A"),
    ):
        rows = report["histories"][name]["evaluations"]
        axis.plot(
            [row["step"] for row in rows],
            [100 * row["success_rate"] for row in rows],
            marker="o",
            label=label,
            color=color,
        )
    axis.set_ylim(-2, 102)
    axis.set_xlabel("环境交互步数")
    axis.set_ylabel("8×8 成功率")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, output / "learning-curves.png")
    variants = list(report["variants"])
    values = [100 * report["variants"][name]["best_success_rate"] for name in variants]
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(["直接", "访问计数", "课程"], values, color=["#64748B", "#F97316", "#16A34A"])
    axis.bar_label(bars, fmt="%.0f%%", padding=3)
    axis.set_ylim(0, 105)
    axis.set_ylabel("最佳成功率")
    axis.grid(axis="y", alpha=0.2)
    save(figure, output / "best-results.png")
    figure, axis = plt.subplots(figsize=(9, 5))
    for name, label, color in (
        ("direct-sparse-8x8", "直接", "#64748B"),
        ("count-bonus-8x8", "访问计数", "#F97316"),
        ("curriculum-8x8", "课程", "#16A34A"),
    ):
        rows = report["histories"][name]["updates"]
        axis.plot(
            [row["step"] for row in rows],
            [row["entropy"] for row in rows],
            label=label,
            color=color,
        )
    axis.set_title("课程切换时策略仍保持探索")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, output / "entropy.png")
    cover(22, "先学会动作链，再扩大地图", output / "conclusion.png")


def lesson23(report: dict[str, Any], output: Path) -> None:
    cover(23, "示范是起点，不是免疫牌", output / "cover.png")
    bc = report["behavior_cloning"]["history"]
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(
        [row["epoch"] for row in bc],
        [100 * row["validation_accuracy"] for row in bc],
        label="动作分类准确率",
        color="#64748B",
    )
    axis.plot(
        [row["epoch"] for row in bc],
        [100 * row["success_rate"] for row in bc],
        marker="o",
        label="闭环成功率",
        color="#2563EB",
    )
    axis.set_ylim(0, 105)
    axis.set_xlabel("行为克隆轮次")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, output / "bc-curves.png")
    rows = [
        report["naive_finetune"]["evaluations"][-1],
        report["reference_kl_finetune"]["evaluations"][-1],
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        ["普通 PPO 微调", "参考 KL 微调"],
        [100 * row["success_rate"] for row in rows],
        color=["#EF4444", "#16A34A"],
    )
    axis.bar_label(bars, fmt="%.0f%%", padding=3)
    axis.set_ylim(0, 100)
    axis.set_ylabel("成功率")
    axis.grid(axis="y", alpha=0.2)
    save(figure, output / "finetune-comparison.png")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for item, label, color in (
        (report["naive_finetune"], "普通 PPO", "#EF4444"),
        (report["reference_kl_finetune"], "参考 KL", "#16A34A"),
    ):
        rows = item["updates"]
        axes[0].plot(
            [row["step"] for row in rows],
            [row["entropy"] for row in rows],
            label=label,
            color=color,
        )
        axes[1].plot(
            [row["step"] for row in rows],
            [row["approx_kl"] for row in rows],
            label=label,
            color=color,
        )
    axes[0].set_title("策略熵")
    axes[1].set_title("近似 KL")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    save(figure, output / "finetune-diagnostics.png")
    cover(23, "防止灾难性遗忘", output / "conclusion.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate advanced DoorKey lesson figures")
    parser.add_argument("--lesson", type=int, choices=(21, 22, 23), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if args.lesson == 21:
        lesson21(report, args.output_dir)
    elif args.lesson == 22:
        lesson22(report, args.output_dir)
    else:
        lesson23(report, args.output_dir)


if __name__ == "__main__":
    main()

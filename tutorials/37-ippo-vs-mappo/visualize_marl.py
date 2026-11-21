"""Generate publication figures for multi-agent chapters 32-39."""

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
    32: "一个奖励，多个智能体该怎么分工",
    33: "三个 DQN 一起训练，环境为什么一直在变",
    34: "多个智能体应该共用一个大脑吗",
    35: "团队得分应该由谁负责",
    36: "VDN 不够灵活时，QMIX 如何混合价值",
    37: "集中训练，分散执行",
    38: "没有通信，局部观测下如何配合",
    39: "训练成功不等于团队可靠",
}


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def cover(lesson: int, subtitle: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.75), facecolor="#111827")
    axis.axis("off")
    axis.text(0.06, 0.80, f"RL 强化学习从小白到老鸟（{lesson}）", color="#A5B4FC", fontsize=18)
    axis.text(0.06, 0.53, TITLES[lesson], color="white", fontsize=32, weight="bold")
    axis.text(0.06, 0.32, subtitle, color="#FBBF24", fontsize=22, weight="bold")
    points = np.asarray([[0.78, 0.70], [0.88, 0.47], [0.74, 0.28]])
    for first in range(3):
        for second in range(first + 1, 3):
            axis.plot(*zip(points[first], points[second], strict=True), color="#475569", lw=4)
    axis.scatter(points[:, 0], points[:, 1], s=420, color=["#38BDF8", "#F97316", "#22C55E"])
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    save(figure, path)


def bars(labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    colors = ["#64748B", "#0EA5E9", "#F97316", "#16A34A", "#8B5CF6"]
    rectangles = axis.bar(labels, values, color=colors[: len(labels)])
    axis.bar_label(rectangles, fmt="%.2f", padding=3)
    axis.set_title(title, fontsize=16, weight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.2)
    save(figure, path)


def history_of(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    if "history" in payload:
        return payload["history"]["evaluations"]
    return None


def generic_training(lesson: int, report: dict[str, Any], output: Path) -> None:
    cover(lesson, "相同预算下只改变一个协作机制", output / "cover.png")
    variants = {
        name: row for name, row in report["variants"].items() if history_of(row) is not None
    }
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for name, payload in variants.items():
        rows = history_of(payload)
        assert rows is not None
        axis.plot(
            [row["step"] for row in rows],
            [row["average_team_return"] for row in rows],
            marker="o",
            label=name.replace("_", " "),
        )
    axis.set_xlabel("环境步数")
    axis.set_ylabel("团队回报")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, output / "learning-curves.png")
    names = list(variants)
    bars(
        [name.replace("_", "\n") for name in names],
        [variants[name]["best"]["average_team_return"] for name in names],
        "最佳固定评估团队回报",
        "团队回报",
        output / "best-returns.png",
    )
    bars(
        [name.replace("_", "\n") for name in names],
        [100 * variants[name]["best"]["average_coverage"] for name in names],
        "最终目标覆盖率",
        "覆盖率（%）",
        output / "coverage.png",
    )


def lesson32(report: dict[str, Any], output: Path) -> None:
    cover(32, "Simple Spread：共享奖励与目标分配", output / "cover.png")
    names = ["random", "nearest", "assignment"]
    labels = ["随机", "各追最近", "全局匹配"]
    bars(
        labels,
        [report["summaries"][name]["average_team_return"] for name in names],
        "规则基线团队回报",
        "团队回报",
        output / "baseline-returns.png",
    )
    bars(
        labels,
        [100 * report["summaries"][name]["average_coverage"] for name in names],
        "终局目标覆盖率",
        "覆盖率（%）",
        output / "coverage.png",
    )
    cover(32, "目标分配正确，不代表低层控制已经解决", output / "conclusion.png")


def lesson38(report: dict[str, Any], output: Path) -> None:
    cover(38, "学到消息，不等于用到了消息", output / "cover.png")
    names = list(report["results"])
    labels = ["完整消息", "丢失一半", "全部切断"]
    bars(
        labels,
        [report["results"][name]["summary"]["average_team_return"] for name in names],
        "消息消融后的团队回报",
        "团队回报",
        output / "message-ablation.png",
    )
    bars(
        labels,
        [100 * report["results"][name]["summary"]["average_coverage"] for name in names],
        "消息消融后的目标覆盖率",
        "覆盖率（%）",
        output / "coverage.png",
    )
    cover(38, "只有切断消息后行为改变，才能证明通信被使用", output / "conclusion.png")


def lesson39(report: dict[str, Any], output: Path) -> None:
    cover(39, "未见种子、观测噪声与单体失效", output / "cover.png")
    agents = list(report["results"])
    conditions = ["clean", "noise", "agent_failure"]
    matrix = np.asarray(
        [
            [
                report["results"][agent]["conditions"][condition]["summary"]["average_team_return"]
                for condition in conditions
            ]
            for agent in agents
        ]
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    image = axis.imshow(matrix, cmap="RdYlGn", aspect="auto")
    axis.set_xticks(range(3), ["干净", "观测噪声", "单体失效"])
    axis.set_yticks(range(len(agents)), [name.upper() for name in agents])
    for row in range(len(agents)):
        for column in range(3):
            axis.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="团队回报")
    axis.set_title("可靠性条件矩阵", fontsize=16, weight="bold")
    save(figure, output / "reliability-heatmap.png")
    clean = [report["results"][name]["conditions"]["clean"] for name in agents]
    bars(
        [name.upper() for name in agents],
        [row["p10_return"] for row in clean],
        "干净条件的 10% 分位回报",
        "团队回报",
        output / "tail-risk.png",
    )
    cover(39, "团队可靠性取决于最脆弱的协作环节", output / "conclusion.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lesson", type=int, choices=range(32, 40), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if args.lesson == 32:
        lesson32(report, args.output_dir)
    elif args.lesson == 38:
        lesson38(report, args.output_dir)
    elif args.lesson == 39:
        lesson39(report, args.output_dir)
    else:
        generic_training(args.lesson, report, args.output_dir)


if __name__ == "__main__":
    main()

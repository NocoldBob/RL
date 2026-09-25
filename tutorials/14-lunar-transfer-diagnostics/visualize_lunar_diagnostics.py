"""Generate lesson 14 figures from the transfer diagnostic report."""

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
    axis.text(
        0.06,
        0.76,
        "RL 强化学习从小白到老鸟（十四）",
        transform=axis.transAxes,
        color="#93C5FD",
        fontsize=18,
    )
    axis.text(
        0.06,
        0.53,
        "代码能运行",
        transform=axis.transAxes,
        color="white",
        fontsize=38,
        weight="bold",
    )
    axis.text(
        0.06,
        0.34,
        "为什么月球还是落不下来？",
        transform=axis.transAxes,
        color="#FDBA74",
        fontsize=30,
        weight="bold",
    )
    axis.plot([0.70, 0.90], [0.70, 0.70], transform=axis.transAxes, color="#22C55E", linewidth=7)
    axis.plot([0.72, 0.88], [0.67, 0.25], transform=axis.transAxes, color="#EF4444", linewidth=5)
    axis.scatter(
        [0.88], [0.25], transform=axis.transAxes, s=500, marker="x", color="#EF4444", linewidths=6
    )
    axis.text(
        0.80,
        0.78,
        "训练回报改善",
        transform=axis.transAxes,
        color="#BBF7D0",
        fontsize=15,
        ha="center",
    )
    axis.text(
        0.80,
        0.14,
        "最终仍然坠毁",
        transform=axis.transAxes,
        color="#FCA5A5",
        fontsize=15,
        ha="center",
    )
    save(figure, path)


def plot_transfer_curves(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for key, label, color in (
        ("shared_encoder_ppo", "共享编码器", "#64748B"),
        ("separate_encoder_ppo", "独立编码器", "#2563EB"),
    ):
        rows = report[key]["evaluations"]
        steps = [row["step"] for row in rows]
        axes[0].plot(
            steps, [row["average_return"] for row in rows], marker="o", label=label, color=color
        )
        axes[1].plot(
            steps, [100 * row["landing_rate"] for row in rows], marker="o", label=label, color=color
        )
    axes[0].axhline(200, color="#16A34A", linestyle="--", label="解决线 200")
    axes[0].set_title("平均回报")
    axes[1].set_title("安全着陆率")
    axes[1].set_ylim(-3, 103)
    for axis in axes:
        axis.set_xlabel("环境交互步数")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("增加训练步数没有自动解决迁移失败", fontsize=17, weight="bold")
    save(figure, path)


def plot_collapse(report: dict[str, Any], path: Path) -> None:
    rows = report["separate_encoder_ppo"]["evaluations"]
    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(
        steps,
        [100 * row["landing_rate"] for row in rows],
        marker="o",
        color="#2563EB",
        label="着陆率",
    )
    axes[0].axvline(200000, color="#16A34A", linestyle="--", label="最佳检查点")
    axes[0].set_ylabel("着陆率（%）")
    axes[1].plot(
        steps, [row["entropy"] for row in rows], marker="o", color="#F97316", label="策略熵"
    )
    axes[1].plot(
        steps,
        [100 * row["clip_fraction"] for row in rows],
        marker="s",
        color="#A855F7",
        label="裁剪比例 ×100",
    )
    for axis in axes:
        axis.set_xlabel("环境交互步数")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("最佳策略出现后仍会被后续更新破坏", fontsize=17, weight="bold")
    save(figure, path)


def plot_probe_matrix(report: dict[str, Any], path: Path) -> None:
    labels = ["PPO\n共享网络", "PPO\n独立网络", "SAC\n缩放奖励", "SAC\n原始奖励"]
    returns = [
        -62.9,
        10.2,
        report["sac_probes"][0]["best_average_return"],
        report["sac_probes"][1]["best_average_return"],
    ]
    colors = ["#64748B", "#2563EB", "#F97316", "#EA580C"]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(labels, returns, color=colors)
    axis.bar_label(bars, fmt="%.1f", padding=4)
    axis.axhline(200, color="#16A34A", linestyle="--", label="解决线 200")
    axis.axhline(0, color="#334155", linewidth=1)
    axis.set_ylabel("探测阶段最佳平均回报")
    axis.set_title("代码迁移完成，不等于任务已经学会", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_transfer_curves(report, output_dir / "transfer-curves.png")
    plot_collapse(report, output_dir / "policy-collapse.png")
    plot_probe_matrix(report, output_dir / "probe-results.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LunarLander transfer diagnostics")
    parser.add_argument(
        "--input", type=Path, default=Path("docs/experiments/14-lunar-transfer-diagnostics.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/csdn-14"))
    args = parser.parse_args()
    generate_assets(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()

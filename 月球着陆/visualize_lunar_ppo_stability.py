"""Generate lesson 15 figures from the PPO stability report."""

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
    figure, axis = plt.subplots(figsize=(12, 6.75), facecolor="#0F172A")
    axis.axis("off")
    axis.text(0.06, 0.78, "RL 强化学习从小白到老鸟（十五）", color="#93C5FD", fontsize=18)
    axis.text(0.06, 0.53, "一次成功不算稳定", color="white", fontsize=38, weight="bold")
    axis.text(0.06, 0.34, "用三随机种子检验 PPO", color="#FBBF24", fontsize=29, weight="bold")
    axis.plot([0.72, 0.92], [0.72, 0.72], color="#22C55E", linewidth=8)
    axis.plot([0.72, 0.92], [0.49, 0.32], color="#EF4444", linewidth=6)
    axis.plot([0.72, 0.92], [0.28, 0.20], color="#F97316", linewidth=6)
    axis.text(0.82, 0.80, "90%", color="#86EFAC", fontsize=18, ha="center")
    axis.text(0.82, 0.10, "仍有两个种子失败", color="#FCA5A5", fontsize=15, ha="center")
    save(figure, path)


def plot_learning_curves(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {7: "#F97316", 42: "#2563EB", 2026: "#7C3AED"}
    histories = report["variants"]["stabilized"]["histories"]
    for seed, history in histories.items():
        rows = history["evaluations"]
        steps = [row["step"] for row in rows]
        axes[0].plot(
            steps,
            [row["average_return"] for row in rows],
            marker="o",
            color=colors[int(seed)],
            label=f"seed={seed}",
        )
        axes[1].plot(
            steps,
            [100 * row["landing_rate"] for row in rows],
            marker="o",
            color=colors[int(seed)],
            label=f"seed={seed}",
        )
    axes[0].set_title("平均回报")
    axes[1].set_title("安全着陆率")
    axes[1].axhline(60, linestyle="--", color="#16A34A", label="筛选线 60%")
    axes[1].set_ylim(-3, 103)
    for axis in axes:
        axis.set_xlabel("环境交互步数")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("相同代码，不同随机种子走向完全不同", fontsize=17, weight="bold")
    save(figure, path)


def plot_variant_comparison(report: dict[str, Any], path: Path) -> None:
    seeds = report["seeds"]
    old = {row["seed"]: row for row in report["variants"]["conservative"]["runs"]}
    new = {row["seed"]: row for row in report["variants"]["stabilized"]["runs"]}
    x = range(len(seeds))
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.bar(
        [value - 0.2 for value in x],
        [100 * old[s]["final_landing_rate"] for s in seeds],
        width=0.4,
        label="5 轮更新，std=1.0",
        color="#94A3B8",
    )
    bars = axis.bar(
        [value + 0.2 for value in x],
        [100 * new[s]["final_landing_rate"] for s in seeds],
        width=0.4,
        label="10 轮+价值裁剪，std≈0.61",
        color="#2563EB",
    )
    axis.bar_label(bars, fmt="%.0f%%", padding=3)
    axis.set_xticks(list(x), [f"seed={seed}" for seed in seeds])
    axis.set_ylabel("20 万步着陆率（%）")
    axis.set_ylim(0, 100)
    axis.set_title("改进提高了上限，但没有消除种子敏感性", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(figure, path)


def plot_diagnostics(report: dict[str, Any], path: Path) -> None:
    rows = report["variants"]["stabilized"]["runs"]
    labels = [str(row["seed"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(labels, [row["mean_approx_kl"] for row in rows], color="#0EA5E9")
    axes[0].axhline(0.015, linestyle="--", color="#EF4444", label="目标 KL")
    axes[0].set_title("平均近似 KL")
    axes[0].legend()
    axes[1].bar(labels, [row["final_explained_variance"] for row in rows], color="#8B5CF6")
    axes[1].set_title("最终价值解释率")
    for axis in axes:
        axis.set_xlabel("随机种子")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("更新幅度相近，结果仍可能分化", fontsize=16, weight="bold")
    save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_learning_curves(report, output_dir / "learning-curves.png")
    plot_variant_comparison(report, output_dir / "variant-comparison.png")
    plot_diagnostics(report, output_dir / "diagnostics.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lesson 15 figures")
    parser.add_argument(
        "--input", type=Path, default=Path("docs/experiments/15-lunar-ppo-stability.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/csdn-15"))
    args = parser.parse_args()
    generate_assets(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()

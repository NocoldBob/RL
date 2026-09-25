"""Generate lesson 24 generalization figures."""

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
    axis.text(0.06, 0.78, "RL 强化学习从小白到老鸟（二十四）", color="#93C5FD", fontsize=18)
    axis.text(0.06, 0.52, "换一张地图", color="white", fontsize=43, weight="bold")
    axis.text(0.06, 0.32, "策略还能认出钥匙和门吗？", color="#FBBF24", fontsize=30, weight="bold")
    axis.text(0.80, 0.58, "5 → 8 → 16", color="#60A5FA", fontsize=32, weight="bold", ha="center")
    axis.text(0.80, 0.28, "泛化不是训练成功的附赠品", color="#C4B5FD", fontsize=17, ha="center")
    save(figure, path)


def plot_heatmap(report: dict[str, Any], path: Path) -> None:
    variants = list(dict.fromkeys(row["variant"] for row in report["results"]))
    sizes = [5, 6, 8, 16]
    matrix = np.asarray(
        [
            [
                100
                * next(
                    row["success_rate"]
                    for row in report["results"]
                    if row["variant"] == variant and row["size"] == size
                )
                for size in sizes
            ]
            for variant in variants
        ]
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    image = axis.imshow(matrix, cmap="YlGn", vmin=0, vmax=100)
    axis.set_xticks(range(len(sizes)), [f"{size}×{size}" for size in sizes])
    axis.set_yticks(range(len(variants)), variants)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.0f}%", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="成功率（%）")
    axis.set_title("训练尺寸、评估尺寸与成功率", fontsize=16, weight="bold")
    save(figure, path)


def plot_steps(report: dict[str, Any], path: Path) -> None:
    sizes = [5, 6, 8, 16]
    figure, axis = plt.subplots(figsize=(10, 5))
    for variant in ("curriculum-8x8", "behavior-cloning", "bc-reference-kl"):
        rows = [row for row in report["results"] if row["variant"] == variant]
        axis.plot(sizes, [row["average_steps"] for row in rows], marker="o", label=variant)
    axis.set_xticks(sizes, [f"{size}×{size}" for size in sizes])
    axis.set_ylabel("平均回合步数")
    axis.set_title("成功率之外，还要看策略是否陷入超时", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(alpha=0.2)
    save(figure, path)


def plot_conclusion(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.2))
    axis.axis("off")
    axis.text(
        0.5, 0.72, "训练成功 ≠ 泛化成功", ha="center", fontsize=28, weight="bold", color="#1D4ED8"
    )
    axis.text(0.5, 0.48, "随机种子、地图尺寸、观测别名", ha="center", fontsize=20, color="#475569")
    axis.text(
        0.5, 0.27, "必须分别设计评估集", ha="center", fontsize=24, weight="bold", color="#EA580C"
    )
    save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_heatmap(report, output_dir / "generalization-heatmap.png")
    plot_steps(report, output_dir / "episode-lengths.png")
    plot_conclusion(output_dir / "conclusion.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lesson 24 figures")
    parser.add_argument(
        "--input", type=Path, default=Path("docs/experiments/24-doorkey-generalization.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/csdn-24"))
    args = parser.parse_args()
    generate_assets(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()

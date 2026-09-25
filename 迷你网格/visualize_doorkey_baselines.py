"""Generate lesson 18 DoorKey baseline figures."""

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
    axis.text(0.06, 0.78, "RL 强化学习从小白到老鸟（十八）", color="#93C5FD", fontsize=18)
    axis.text(0.06, 0.52, "门就在眼前", color="white", fontsize=42, weight="bold")
    axis.text(0.06, 0.32, "为什么没有钥匙还是过不去？", color="#FBBF24", fontsize=29, weight="bold")
    axis.text(0.75, 0.60, "KEY", color="#FACC15", fontsize=38, weight="bold", ha="center")
    axis.text(0.89, 0.60, "DOOR", color="#A78BFA", fontsize=32, weight="bold", ha="center")
    axis.annotate(
        "",
        xy=(0.85, 0.60),
        xytext=(0.80, 0.60),
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 3},
    )
    axis.text(0.82, 0.22, "稀疏奖励 · 部分可观测", color="#C4B5FD", fontsize=17, ha="center")
    save(figure, path)


def plot_success(report: dict[str, Any], path: Path) -> None:
    labels = ["随机", "只向前", "搜索专家"]
    policies = ["random", "forward", "expert"]
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for offset, size, color in ((-0.18, 5, "#2563EB"), (0.18, 8, "#F97316")):
        values = [
            100 * report["groups"][f"{size}x{size}-{policy}"]["success_rate"] for policy in policies
        ]
        bars = axis.bar(x + offset, values, width=0.36, label=f"{size}×{size}", color=color)
        axis.bar_label(bars, fmt="%.0f%%", padding=3)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 108)
    axis.set_ylabel("成功率")
    axis.set_title("随机动作偶尔成功，搜索专家稳定完成", fontsize=16, weight="bold")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(figure, path)


def plot_milestones(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for axis, size in zip(axes, (5, 8), strict=True):
        group = report["groups"][f"{size}x{size}-random"]
        values = [100 * group[key] for key in ("key_pickup_rate", "door_open_rate", "success_rate")]
        bars = axis.bar(
            ["拾取钥匙", "打开门", "到达终点"], values, color=["#EAB308", "#8B5CF6", "#16A34A"]
        )
        axis.bar_label(bars, fmt="%.0f%%", padding=3)
        axis.set_title(f"{size}×{size} 随机策略")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("阶段完成率")
    figure.suptitle("真正的瓶颈不是看见钥匙，而是完成动作链", fontsize=16, weight="bold")
    save(figure, path)


def plot_expert_trace(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for axis, size in zip(axes, (5, 8), strict=True):
        trace = report["expert_traces"][f"{size}x{size}-expert"]
        x = trace["x"]
        y = trace["y"]
        axis.plot(x, y, marker="o", color="#2563EB")
        axis.scatter([x[0]], [y[0]], s=120, color="#16A34A", label="起点")
        axis.scatter([x[-1]], [y[-1]], s=120, color="#EF4444", label="终点前")
        for index, (px, py) in enumerate(zip(x, y, strict=True)):
            axis.text(px + 0.05, py + 0.05, str(index), fontsize=8)
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_title(f"{size}×{size} 最短动作轨迹")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("规则专家为后续模仿学习提供可验证示范", fontsize=16, weight="bold")
    save(figure, path)


def generate_assets(report: dict[str, Any], output_dir: Path) -> None:
    plot_cover(output_dir / "cover.png")
    plot_success(report, output_dir / "success-rates.png")
    plot_milestones(report, output_dir / "random-milestones.png")
    plot_expert_trace(report, output_dir / "expert-traces.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lesson 18 figures")
    parser.add_argument(
        "--input", type=Path, default=Path("docs/experiments/18-doorkey-baselines.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/csdn-18"))
    args = parser.parse_args()
    generate_assets(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()

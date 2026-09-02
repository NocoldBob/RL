"""Create article figures from the continuous-control baseline report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mountain_car_baselines import ENV_ID, POLICY_LABELS, POLICY_NAMES

COLORS = {
    "zero": "#6b7280",
    "random": "#e76f51",
    "bang_bang": "#2a9d8f",
    "smooth_momentum": "#457b9d",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#f7f8fa",
        }
    )


def save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def plot_cover(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.3))
    figure.patch.set_facecolor("#111827")
    axis.set_facecolor("#111827")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.07, 0.84, "CONTINUOUS CONTROL", color="#6ee7b7", fontsize=16)
    axis.text(
        0.07,
        0.62,
        "THROTTLE IS\nNOT A SWITCH",
        color="#f9fafb",
        fontsize=38,
        weight="bold",
        va="center",
    )
    axis.text(0.07, 0.16, "LESSON 09  /  MOUNTAIN CAR", color="#cbd5e1", fontsize=15)
    x = np.linspace(-1.1, 0.75, 300)
    y = 0.18 + 0.28 * (np.sin(3 * x) + 1.0) / 2.0
    x_plot = 0.61 + (x - x.min()) / (x.max() - x.min()) * 0.34
    axis.plot(x_plot, y, color="#94a3b8", linewidth=4)
    car_index = 95
    axis.scatter(
        [x_plot[car_index]],
        [y[car_index] + 0.025],
        s=280,
        color="#f97316",
        edgecolor="#f9fafb",
        linewidth=2,
        zorder=3,
    )
    axis.annotate(
        "",
        xy=(x_plot[car_index] - 0.07, y[car_index] + 0.025),
        xytext=(x_plot[car_index] - 0.01, y[car_index] + 0.025),
        arrowprops={"arrowstyle": "-|>", "color": "#6ee7b7", "lw": 3},
    )
    save_figure(figure, output)


def plot_environment(output: Path, seed: int) -> None:
    env = gym.make(ENV_ID, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=seed)
        frame = env.render()
    finally:
        env.close()
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.imshow(frame)
    axis.axis("off")
    axis.set_title(
        f"MountainCarContinuous-v0   position={observation[0]:.3f}   velocity={observation[1]:.3f}",
        pad=12,
        weight="bold",
    )
    save_figure(figure, output)


def plot_action_space(output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    discrete = np.asarray([-1.0, 0.0, 1.0])
    axes[0].hlines(0, -1.1, 1.1, color="#cbd5e1", linewidth=2)
    axes[0].scatter(discrete, np.zeros(3), s=180, color="#e76f51", zorder=3)
    for value, label in zip(discrete, ("left", "coast", "right"), strict=True):
        axes[0].text(value, 0.12, label, ha="center")
    axes[0].set_title("Discrete actions: three choices", weight="bold")
    axes[0].set_xlim(-1.15, 1.15)
    axes[0].set_ylim(-0.25, 0.3)
    axes[0].set_yticks([])
    axes[0].set_xlabel("action")

    actions = np.linspace(-1.0, 1.0, 201)
    costs = 0.1 * actions**2
    axes[1].plot(actions, costs, color="#457b9d", linewidth=3)
    axes[1].fill_between(actions, 0, costs, color="#bfdbfe", alpha=0.7)
    axes[1].set_title("Continuous throttle and action cost", weight="bold")
    axes[1].set_xlabel("throttle in [-1, 1]")
    axes[1].set_ylabel("cost per step: 0.1 x action^2")
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "A continuous action is a value, not an action number", fontsize=17, weight="bold"
    )
    figure.tight_layout()
    save_figure(figure, output)


def _bar_chart(
    axis: plt.Axes,
    values: list[float],
    title: str,
    formatter: str = ".1f",
) -> None:
    bars = axis.bar(
        range(len(POLICY_NAMES)),
        values,
        color=[COLORS[name] for name in POLICY_NAMES],
        width=0.68,
    )
    axis.axhline(0, color="#9ca3af", linewidth=1)
    axis.set_title(title, weight="bold")
    axis.set_xticks(range(len(POLICY_NAMES)), ["Zero", "Random", "Full", "Smooth"])
    axis.grid(axis="y", alpha=0.18)
    axis.margins(y=0.12)
    for bar, value in zip(bars, values, strict=True):
        offset = 4 if value >= 0 else -14
        axis.annotate(
            format(value, formatter),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )


def plot_results(report: dict[str, Any], output: Path) -> None:
    aggregate = report["aggregate"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    _bar_chart(
        axes[0, 0],
        [aggregate[name]["average_return"] for name in POLICY_NAMES],
        "Average return",
    )
    _bar_chart(
        axes[0, 1],
        [100 * aggregate[name]["success_rate"] for name in POLICY_NAMES],
        "Success rate (%)",
        ".0f",
    )
    _bar_chart(
        axes[1, 0],
        [aggregate[name]["average_steps"] for name in POLICY_NAMES],
        "Average episode steps",
        ".0f",
    )
    _bar_chart(
        axes[1, 1],
        [aggregate[name]["average_action_cost"] for name in POLICY_NAMES],
        "Average action cost",
    )
    figure.suptitle(
        f"Four baselines, {report['experiment']['episodes_per_policy']} matched starts each",
        fontsize=17,
        weight="bold",
    )
    figure.tight_layout()
    save_figure(figure, output)


def plot_trajectories(report: dict[str, Any], output: Path) -> None:
    traces = report["traces"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for policy in POLICY_NAMES:
        trace = traces[policy]
        color = COLORS[policy]
        label = POLICY_LABELS[policy]
        axes[0, 0].plot(trace["position"], color=color, label=label, linewidth=1.8)
        axes[0, 1].plot(trace["velocity"], color=color, label=label, linewidth=1.5)
        axes[1, 0].plot(trace["action"], color=color, label=label, linewidth=1.3)
        axes[1, 1].plot(
            np.cumsum(trace["reward"]),
            color=color,
            label=label,
            linewidth=1.8,
        )
    axes[0, 0].axhline(0.45, color="#111827", linestyle="--", linewidth=1, label="Goal")
    axes[0, 0].set_title("Position", weight="bold")
    axes[0, 1].set_title("Velocity", weight="bold")
    axes[1, 0].set_title("Throttle", weight="bold")
    axes[1, 1].set_title("Cumulative reward", weight="bold")
    for axis in axes.flat:
        axis.set_xlabel("step")
        axis.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    figure.suptitle(
        f"Same initial state, seed={report['experiment']['trace_seed']}",
        fontsize=17,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_figure(figure, output)


def generate_assets(report: dict[str, Any], assets_dir: Path) -> None:
    configure_style()
    plot_cover(assets_dir / "cover.png")
    plot_environment(assets_dir / "environment.png", report["experiment"]["trace_seed"])
    plot_action_space(assets_dir / "action-space.png")
    plot_results(report, assets_dir / "baseline-results.png")
    plot_trajectories(report, assets_dir / "trajectories.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Lesson 9 article figures")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("docs/experiments/09-continuous-baselines.json"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("docs/assets/csdn-09"),
    )
    arguments = parser.parse_args()
    report = json.loads(arguments.input_json.read_text(encoding="utf-8"))
    generate_assets(report, arguments.assets_dir)


if __name__ == "__main__":
    main()

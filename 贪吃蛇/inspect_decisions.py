"""Inspect trained Snake policies on the same hand-authored states."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from benchmark import load_actor_critic, load_dqn, load_ppo
from dqn import DuelingQNetwork
from environment import SnakeEnv

ACTION_NAMES = ("left", "right", "straight")
ACTION_SHORT_NAMES = ("L", "R", "S")
ACTION_COLORS = ("#e76f51", "#2a9d8f", "#457b9d")
GRID_SIZE = 6


@dataclass(frozen=True)
class Scenario:
    """A valid, fixed Snake state used for cross-model inspection."""

    name: str
    title: str
    description: str
    snake: tuple[tuple[int, int], ...]
    food: tuple[int, int]
    direction: tuple[int, int]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    checkpoint: Path


def build_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            name="food_ahead",
            title="Food ahead",
            description="Open board with food two cells in front of the head.",
            snake=((3, 3),),
            food=(3, 5),
            direction=(0, 1),
        ),
        Scenario(
            name="wall_ahead",
            title="Wall ahead",
            description="Moving straight collides with the right boundary.",
            snake=((3, 5), (3, 4)),
            food=(0, 0),
            direction=(0, 1),
        ),
        Scenario(
            name="body_turn",
            title="Body turn",
            description="A bent body makes the local geometry visible to the policy.",
            snake=((2, 2), (2, 1), (3, 1)),
            food=(4, 2),
            direction=(0, 1),
        ),
        Scenario(
            name="narrow_escape",
            title="Narrow escape",
            description="Turning right hits the body; left and straight remain immediately safe.",
            snake=((1, 1), (1, 2), (2, 2), (2, 1), (3, 1)),
            food=(0, 5),
            direction=(-1, 0),
        ),
    )


def scenario_environment(scenario: Scenario) -> SnakeEnv:
    env = SnakeEnv(grid_size=GRID_SIZE, end_score=6, max_steps=100)
    env.snake = [np.asarray(position, dtype=np.int64) for position in scenario.snake]
    env.food_pos = np.asarray(scenario.food, dtype=np.int64)
    env.current_direction = np.asarray(scenario.direction, dtype=np.int64)
    env.score = max(0, len(scenario.snake) - 1)
    env.step_count = 0
    env.game_over = False
    return env


def scenario_observation(scenario: Scenario) -> np.ndarray:
    env = scenario_environment(scenario)
    try:
        return env._get_observation()
    finally:
        env.close()


def scenario_immediate_safety(scenario: Scenario) -> tuple[bool, ...]:
    env = scenario_environment(scenario)
    try:
        head = env.snake[0]
        return tuple(
            not env._is_collision(head + env._get_direction(action))
            for action in range(env.action_space.n)
        )
    finally:
        env.close()


def scenario_teacher_action(scenario: Scenario) -> int:
    env = scenario_environment(scenario)
    try:
        return env.teacher_action()
    finally:
        env.close()


def centered_advantages(advantages: np.ndarray) -> np.ndarray:
    return advantages - advantages.mean()


def _top_margin(scores: np.ndarray) -> float:
    ordered = np.sort(scores)
    return float(ordered[-1] - ordered[-2])


def inspect_actor_critic(
    checkpoint: Path,
    observations: list[np.ndarray],
    device: torch.device,
) -> list[dict[str, Any]]:
    model = load_actor_critic(checkpoint, device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for observation in observations:
            state = torch.from_numpy(observation).unsqueeze(0).to(device=device)
            logits, value = model(state)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            rows.append(
                {
                    "score_type": "probability",
                    "scores": probabilities.tolist(),
                    "selected_action": int(probabilities.argmax()),
                    "margin": _top_margin(probabilities),
                    "state_value": float(value.item()),
                }
            )
    return rows


def inspect_ppo(
    checkpoint: Path,
    observations: list[np.ndarray],
    device: torch.device,
) -> list[dict[str, Any]]:
    agent = load_ppo(checkpoint, device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for observation in observations:
            state = torch.from_numpy(observation).unsqueeze(0).to(device=device)
            logits, value = agent.model(state)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            rows.append(
                {
                    "score_type": "probability",
                    "scores": probabilities.tolist(),
                    "selected_action": int(probabilities.argmax()),
                    "margin": _top_margin(probabilities),
                    "state_value": float(value.item()),
                }
            )
    return rows


def inspect_dqn(
    checkpoint: Path,
    observations: list[np.ndarray],
    device: torch.device,
) -> list[dict[str, Any]]:
    agent = load_dqn(checkpoint, device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for observation in observations:
            state = torch.from_numpy(observation).unsqueeze(0).to(device=device)
            q_values = agent.online(state).squeeze(0).cpu().numpy()
            row: dict[str, Any] = {
                "score_type": "q_value",
                "scores": q_values.tolist(),
                "selected_action": int(q_values.argmax()),
                "margin": _top_margin(q_values),
            }
            if isinstance(agent.online, DuelingQNetwork):
                value, advantages = agent.online.decompose(state)
                raw_advantages = advantages.squeeze(0).cpu().numpy()
                row.update(
                    {
                        "state_value": float(value.item()),
                        "raw_advantages": raw_advantages.tolist(),
                        "centered_advantages": centered_advantages(raw_advantages).tolist(),
                    }
                )
            rows.append(row)
    return rows


def inspect_models(
    model_specs: tuple[ModelSpec, ...],
    scenarios: tuple[Scenario, ...],
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    observations = [scenario_observation(scenario) for scenario in scenarios]
    results: dict[str, list[dict[str, Any]]] = {}
    for spec in model_specs:
        if spec.family == "actor_critic":
            rows = inspect_actor_critic(spec.checkpoint, observations, device)
        elif spec.family == "ppo":
            rows = inspect_ppo(spec.checkpoint, observations, device)
        else:
            rows = inspect_dqn(spec.checkpoint, observations, device)
        results[spec.name] = rows
    return results


def build_report(
    model_specs: tuple[ModelSpec, ...],
    scenarios: tuple[Scenario, ...],
    decisions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    scenario_rows: list[dict[str, Any]] = []
    safe_selection_count = 0
    selection_count = 0
    for scenario_index, scenario in enumerate(scenarios):
        safety = scenario_immediate_safety(scenario)
        selected_actions = [
            int(decisions[spec.name][scenario_index]["selected_action"]) for spec in model_specs
        ]
        model_rows: dict[str, Any] = {}
        for spec, action in zip(model_specs, selected_actions, strict=True):
            row = dict(decisions[spec.name][scenario_index])
            row["selected_action_name"] = ACTION_NAMES[action]
            row["selected_action_immediately_safe"] = safety[action]
            model_rows[spec.name] = row
            safe_selection_count += int(safety[action])
            selection_count += 1
        scenario_rows.append(
            {
                "name": scenario.name,
                "title": scenario.title,
                "description": scenario.description,
                "snake": scenario.snake,
                "food": scenario.food,
                "direction": scenario.direction,
                "immediate_action_safety": dict(zip(ACTION_NAMES, safety, strict=True)),
                "teacher_action": ACTION_NAMES[scenario_teacher_action(scenario)],
                "unique_model_actions": len(set(selected_actions)),
                "models": model_rows,
            }
        )
    return {
        "inspection_scope": {
            "training_seed": 42,
            "grid_size": GRID_SIZE,
            "scenario_count": len(scenarios),
            "model_count": len(model_specs),
            "note": (
                "Policy probabilities and Q values use different scales and must not be "
                "compared numerically across algorithm families."
            ),
        },
        "models": [
            {"name": spec.name, "family": spec.family, "checkpoint": str(spec.checkpoint)}
            for spec in model_specs
        ],
        "summary": {
            "safe_immediate_selections": safe_selection_count,
            "total_selections": selection_count,
            "safe_immediate_selection_rate": safe_selection_count / selection_count,
            "scenarios_with_model_disagreement": sum(
                row["unique_model_actions"] > 1 for row in scenario_rows
            ),
        },
        "scenarios": scenario_rows,
    }


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#ccd2d9",
            "axes.titleweight": "bold",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _draw_board(axis: plt.Axes, scenario: Scenario) -> None:
    axis.set_xlim(0, GRID_SIZE)
    axis.set_ylim(GRID_SIZE, 0)
    axis.set_aspect("equal")
    for position in range(GRID_SIZE + 1):
        axis.axhline(position, color="#dfe3e8", linewidth=0.8, zorder=0)
        axis.axvline(position, color="#dfe3e8", linewidth=0.8, zorder=0)
    food_row, food_column = scenario.food
    axis.scatter(
        food_column + 0.5,
        food_row + 0.5,
        s=220,
        marker="*",
        color="#2a9d8f",
        edgecolor="#1d6f63",
        linewidth=0.8,
        zorder=3,
    )
    for index, (row, column) in enumerate(reversed(scenario.snake)):
        original_index = len(scenario.snake) - index - 1
        color = "#e63946" if original_index == 0 else "#f4a261"
        axis.add_patch(
            plt.Rectangle(
                (column + 0.08, row + 0.08),
                0.84,
                0.84,
                facecolor=color,
                edgecolor="#7f3138" if original_index == 0 else "#a85f28",
                linewidth=1.0,
                zorder=2,
            )
        )
    head_row, head_column = scenario.snake[0]
    direction_row, direction_column = scenario.direction
    axis.arrow(
        head_column + 0.5,
        head_row + 0.5,
        direction_column * 0.34,
        direction_row * 0.34,
        width=0.035,
        head_width=0.18,
        color="#ffffff",
        length_includes_head=True,
        zorder=4,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(scenario.title, pad=8)


def plot_scenarios(scenarios: tuple[Scenario, ...], output: Path) -> None:
    figure, axes = plt.subplots(1, len(scenarios), figsize=(13.5, 3.6))
    for axis, scenario in zip(axes, scenarios, strict=True):
        _draw_board(axis, scenario)
    figure.suptitle("Four fixed states used for decision inspection", fontsize=16, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_selection_matrix(
    model_specs: tuple[ModelSpec, ...],
    scenarios: tuple[Scenario, ...],
    report: dict[str, Any],
    output: Path,
) -> None:
    selected = np.zeros((len(model_specs), len(scenarios)), dtype=np.int64)
    safe = np.zeros_like(selected, dtype=bool)
    for column, scenario_row in enumerate(report["scenarios"]):
        for row, spec in enumerate(model_specs):
            model_row = scenario_row["models"][spec.name]
            selected[row, column] = ACTION_NAMES.index(model_row["selected_action_name"])
            safe[row, column] = model_row["selected_action_immediately_safe"]
    color_map = matplotlib.colors.ListedColormap(ACTION_COLORS)
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    axis.imshow(selected, cmap=color_map, vmin=-0.5, vmax=2.5, aspect="auto")
    for row in range(selected.shape[0]):
        for column in range(selected.shape[1]):
            marker = "safe" if safe[row, column] else "risk"
            axis.text(
                column,
                row,
                f"{ACTION_NAMES[selected[row, column]]}\n{marker}",
                ha="center",
                va="center",
                color="white",
                weight="bold",
                fontsize=9,
            )
    axis.set_xticks(range(len(scenarios)), [scenario.title for scenario in scenarios])
    axis.set_yticks(range(len(model_specs)), [spec.name for spec in model_specs])
    axis.set_title("Greedy action selected by each trained model", fontsize=15, pad=14)
    axis.tick_params(length=0)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_scenario_detail(
    model_specs: tuple[ModelSpec, ...],
    report: dict[str, Any],
    scenario_name: str,
    output: Path,
) -> None:
    scenario_row = next(row for row in report["scenarios"] if row["name"] == scenario_name)
    safety = scenario_row["immediate_action_safety"]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for axis, spec in zip(axes.flat, model_specs, strict=True):
        row = scenario_row["models"][spec.name]
        scores = np.asarray(row["scores"], dtype=np.float64)
        bars = axis.bar(ACTION_SHORT_NAMES, scores, color=ACTION_COLORS, width=0.68)
        for index, bar in enumerate(bars):
            if not safety[ACTION_NAMES[index]]:
                bar.set_hatch("///")
                bar.set_alpha(0.55)
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{scores[index]:.2f}",
                ha="center",
                va="bottom" if scores[index] >= 0 else "top",
                fontsize=8,
            )
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_title(spec.name)
        axis.set_ylabel("probability" if row["score_type"] == "probability" else "Q value")
    figure.suptitle(
        "Narrow escape: action scores (hatched action hits the body)",
        fontsize=16,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_dueling_decomposition(
    model_specs: tuple[ModelSpec, ...],
    scenarios: tuple[Scenario, ...],
    report: dict[str, Any],
    output: Path,
) -> None:
    dueling_specs = [spec for spec in model_specs if "Dueling" in spec.name]
    figure, axes = plt.subplots(len(dueling_specs), len(scenarios), figsize=(13.5, 6.3))
    for row_index, spec in enumerate(dueling_specs):
        for column_index, scenario in enumerate(scenarios):
            axis = axes[row_index, column_index]
            model_row = report["scenarios"][column_index]["models"][spec.name]
            value = float(model_row["state_value"])
            centered = np.asarray(model_row["centered_advantages"], dtype=np.float64)
            q_values = np.asarray(model_row["scores"], dtype=np.float64)
            axis.axhline(value, color="#333333", linestyle="--", linewidth=1.2, label="V(s)")
            axis.bar(
                ACTION_SHORT_NAMES,
                centered,
                bottom=value,
                color=ACTION_COLORS,
                alpha=0.78,
                width=0.68,
                label="centered A",
            )
            axis.scatter(ACTION_SHORT_NAMES, q_values, color="#111111", s=20, zorder=3)
            if row_index == 0:
                axis.set_title(scenario.title)
            if column_index == 0:
                axis.set_ylabel(f"{spec.name}\nvalue scale")
    figure.suptitle(
        "Dueling decomposition: Q(s,a) = V(s) + centered A(s,a)",
        fontsize=16,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_course_map(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.5, 4.2))
    axis.set_xlim(0, 13.5)
    axis.set_ylim(0, 4.2)
    axis.axis("off")
    stages = (
        (0.4, 2.2, "01-03", "Environment\nActor-Critic\nReproducibility", "#457b9d"),
        (3.25, 2.2, "04-05", "DQN and PPO\nTwo learning routes", "#2a9d8f"),
        (6.1, 2.2, "06-07", "Double / Dueling\nControlled comparisons", "#e9c46a"),
        (8.95, 2.2, "08", "Inspect decisions\nClose discrete stage", "#e76f51"),
        (11.8, 2.2, "NEXT", "Continuous control\nLunar Lander", "#6d597a"),
    )
    for index, (x, y, number, label, color) in enumerate(stages):
        axis.add_patch(
            matplotlib.patches.FancyBboxPatch(
                (x, y - 1.25),
                1.8,
                2.5,
                boxstyle="round,pad=0.08,rounding_size=0.08",
                facecolor=color,
                edgecolor="none",
            )
        )
        axis.text(x + 0.9, y + 0.65, number, ha="center", color="white", weight="bold")
        axis.text(x + 0.9, y - 0.25, label, ha="center", va="center", color="white", fontsize=9)
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(stages[index + 1][0] - 0.12, y),
                xytext=(x + 1.92, y),
                arrowprops={"arrowstyle": "->", "color": "#69727d", "lw": 1.8},
            )
    axis.set_title("RL Snake learning path and the next stage", fontsize=17, weight="bold", pad=10)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_cover(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.3), facecolor="#111827")
    axis.set_facecolor("#111827")
    axis.axis("off")
    axis.text(0.07, 0.83, "RL SNAKE", transform=axis.transAxes, color="#6ee7b7", fontsize=18)
    axis.text(
        0.07,
        0.58,
        "SAME STATE\nSIX MODELS",
        transform=axis.transAxes,
        color="white",
        fontsize=34,
        weight="bold",
        linespacing=1.0,
    )
    axis.text(
        0.07,
        0.24,
        "LESSON 08  /  WHAT WOULD THEY DO?",
        transform=axis.transAxes,
        color="#cbd5e1",
        fontsize=13,
    )
    for index, (label, color) in enumerate(zip(ACTION_SHORT_NAMES, ACTION_COLORS, strict=True)):
        x = 0.7 + index * 0.09
        height = (0.34, 0.56, 0.44)[index]
        axis.add_patch(
            matplotlib.patches.FancyBboxPatch(
                (x, 0.2),
                0.055,
                height,
                transform=axis.transAxes,
                boxstyle="round,pad=0.005,rounding_size=0.012",
                facecolor=color,
                edgecolor="none",
            )
        )
        axis.text(x + 0.0275, 0.15, label, transform=axis.transAxes, ha="center", color="white")
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def write_outputs(
    model_specs: tuple[ModelSpec, ...],
    scenarios: tuple[Scenario, ...],
    report: dict[str, Any],
    output_json: Path,
    assets_dir: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _configure_plot_style()
    plot_cover(assets_dir / "cover.png")
    plot_scenarios(scenarios, assets_dir / "fixed-scenarios.png")
    plot_selection_matrix(model_specs, scenarios, report, assets_dir / "selected-actions.png")
    plot_scenario_detail(
        model_specs,
        report,
        scenario_name="narrow_escape",
        output=assets_dir / "narrow-escape-detail.png",
    )
    plot_dueling_decomposition(
        model_specs,
        scenarios,
        report,
        assets_dir / "dueling-decomposition.png",
    )
    plot_course_map(assets_dir / "course-map.png")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect six trained Snake models on four fixed states"
    )
    parser.add_argument(
        "--actor-critic",
        type=Path,
        default=Path("runs/benchmark-ppo/seed-42/actor-critic/checkpoints/latest.pt"),
    )
    parser.add_argument(
        "--ppo",
        type=Path,
        default=Path("runs/benchmark-ppo/seed-42/ppo/checkpoints/latest.pt"),
    )
    benchmark_root = Path("runs/dueling-dqn-benchmark/seed-42")
    parser.add_argument(
        "--dqn",
        type=Path,
        default=benchmark_root / "dqn/checkpoints/latest.pt",
    )
    parser.add_argument(
        "--double-dqn",
        type=Path,
        default=benchmark_root / "double-dqn/checkpoints/latest.pt",
    )
    parser.add_argument(
        "--dueling-dqn",
        type=Path,
        default=benchmark_root / "dueling-dqn/checkpoints/latest.pt",
    )
    parser.add_argument(
        "--dueling-double-dqn",
        type=Path,
        default=benchmark_root / "dueling-double-dqn/checkpoints/latest.pt",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/experiments/08-decision-inspection.json"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("docs/assets/csdn-08"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    model_specs = (
        ModelSpec("Actor-Critic", "actor_critic", arguments.actor_critic),
        ModelSpec("PPO", "ppo", arguments.ppo),
        ModelSpec("DQN", "dqn", arguments.dqn),
        ModelSpec("Double DQN", "dqn", arguments.double_dqn),
        ModelSpec("Dueling DQN", "dqn", arguments.dueling_dqn),
        ModelSpec("Dueling Double DQN", "dqn", arguments.dueling_double_dqn),
    )
    missing = [str(spec.checkpoint) for spec in model_specs if not spec.checkpoint.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required checkpoints are missing:\n"
            f"{formatted}\n"
            "Run benchmark.py and benchmark_dueling_dqn.py first, or pass explicit paths."
        )
    scenarios = build_scenarios()
    device = torch.device(arguments.device)
    decisions = inspect_models(model_specs, scenarios, device)
    report = build_report(model_specs, scenarios, decisions)
    write_outputs(
        model_specs,
        scenarios,
        report,
        arguments.output_json,
        arguments.assets_dir,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

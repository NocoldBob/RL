"""Render four publication figures for lessons 49 through 57."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs" / "experiments"
ASSETS = ROOT / "docs" / "assets"
NAVY, BLUE, ORANGE, GREEN, RED = "#172554", "#2563eb", "#f97316", "#16a34a", "#dc2626"
TITLES = {
    49: ("Learn the Dynamics", "A model turns real transitions into a simulator"),
    50: ("Compounding Error", "One-step accuracy does not guarantee long rollouts"),
    51: ("Random-Shooting MPC", "Plan action sequences, execute only the first action"),
    52: ("CEM Planning", "Concentrate samples around promising action sequences"),
    53: ("Ensemble Uncertainty", "Model disagreement is useful, but not automatically beneficial"),
    54: ("Dyna", "Mix real replay with one-step imagined experience"),
    55: ("MBPO", "Short branched rollouts limit compounding model error"),
    56: ("Offline Model Planning", "Long-horizon optimization can exploit model mistakes"),
    57: ("Reliability Audit", "Test dynamics shift, sensing noise, delay, and tail risk"),
}


def load(name: str) -> dict:
    return json.loads((EXPERIMENTS / name).read_text(encoding="utf-8"))


def save(number: int, filename: str) -> None:
    destination = ASSETS / f"csdn-{number}"
    destination.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(destination / filename, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


def card(number: int, filename: str, title: str, lines: list[str], color: str = BLUE) -> None:
    plt.figure(figsize=(10, 5.6))
    plt.axis("off")
    plt.text(0.05, 0.85, title, fontsize=24, weight="bold", color=NAVY)
    for index, line in enumerate(lines):
        plt.text(0.08, 0.65 - 0.16 * index, line, fontsize=16, color=color if index == 0 else NAVY)
    save(number, filename)


def bars(
    number: int, filename: str, title: str, labels: list[str], values: list[float], ylabel: str
) -> None:
    plt.figure(figsize=(10, 5.6))
    x = np.arange(len(labels))
    items = plt.bar(x, values, color=[BLUE, ORANGE, GREEN, RED] * 3)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title, fontsize=18, weight="bold", color=NAVY)
    plt.grid(axis="y", alpha=0.25)
    for item, value in zip(items, values, strict=True):
        vertical = "bottom" if value >= 0 else "top"
        plt.text(
            item.get_x() + item.get_width() / 2, value, f"{value:.2f}", ha="center", va=vertical
        )
    save(number, filename)


def returns(number: int, filename: str, report: dict) -> None:
    labels = list(report)
    values = [report[name]["summary"]["average_return"] for name in labels]
    bars(number, filename, "Average return on hidden seeds", labels, values, "return")


def conclusion(number: int, headline: str, detail: str) -> None:
    card(number, "conclusion.png", "What the experiment actually says", [headline, detail], GREEN)


def main() -> None:
    for number, (title, subtitle) in TITLES.items():
        card(
            number,
            "cover.png",
            f"Lesson {number}  |  {title}",
            [subtitle, "Pendulum-v1 · model-based RL"],
        )

    dataset = load("49-dynamics-dataset.json")["summary"]
    training = load("49-dynamics-model.json")
    bars(
        49,
        "dataset.png",
        "Versioned real-interaction dataset",
        ["transitions", "episodes"],
        [dataset["transitions"], dataset["episodes"]],
        "count",
    )
    bars(
        49,
        "model-validation.png",
        "Ensemble validation loss",
        [f"model {row['model']}" for row in training["models"]],
        [row["best_validation_loss"] for row in training["models"]],
        "normalized MSE",
    )
    conclusion(
        49,
        "Five models reached validation loss below 0.0028.",
        "The hidden question is whether this remains accurate under recursive rollout.",
    )

    rollout = load("50-rollout-error.json")["horizons"]
    bars(
        50,
        "state-error.png",
        "State error compounds with rollout horizon",
        [str(row["horizon"]) for row in rollout],
        [row["state_rmse"] for row in rollout],
        "state RMSE",
    )
    bars(
        50,
        "reward-error.png",
        "Cumulative reward prediction error",
        [str(row["horizon"]) for row in rollout],
        [row["reward_mae"] for row in rollout],
        "reward MAE",
    )
    conclusion(
        50,
        "State RMSE grew from 0.013 at one step to 3.49 at 50 steps.",
        "Multi-step diagnostics are mandatory before planning with a learned model.",
    )

    experiments = {
        51: ("51-random-shooting-mpc.json", "planning-return.png"),
        52: ("52-cem-planning.json", "cem-return.png"),
        53: ("53-ensemble-uncertainty.json", "ensemble-return.png"),
        54: ("54-dyna.json", "dyna-return.png"),
        55: ("55-mbpo.json", "mbpo-return.png"),
        56: ("56-offline-model-planning.json", "offline-planning-return.png"),
    }
    summaries = {}
    for number, (filename, asset) in experiments.items():
        result = load(filename)["results"]
        if number in (54, 55):
            flattened = {name: row["evaluation"] for name, row in result.items()}
        else:
            flattened = result
        returns(number, asset, flattened)
        labels = list(flattened)
        upright = [100 * flattened[name]["summary"]["upright_fraction"] for name in labels]
        bars(
            number,
            "upright-time.png",
            "Time balanced near upright",
            labels,
            upright,
            "upright time (%)",
        )
        summaries[number] = flattened

    conclusions = {
        51: (
            "Learned-model MPC nearly matched oracle-model MPC.",
            "Replanning every step limited the damage from long open-loop errors.",
        ),
        52: (
            "CEM improved return from about -165 to -112.",
            "More candidates beyond 128 gave no meaningful gain in this run.",
        ),
        53: (
            "The ensemble and uncertainty penalty did not improve this clean test.",
            "Uncertainty machinery must earn its complexity under relevant shifts.",
        ),
        54: (
            "One-step imagination helped gradually; 75% was best here.",
            "Imagined data changed the result without spending more real transitions.",
        ),
        55: (
            "Horizons 1-10 were close; horizon 5 was narrowly best.",
            "Longer model rollouts added no reliable advantage in this budget.",
        ),
        56: (
            "The ensemble beat naive long-horizon planning, but only modestly.",
            "A penalty cannot be declared useful when its measured gain is negligible.",
        ),
    }
    for number, message in conclusions.items():
        conclusion(number, *message)

    audit = load("57-model-reliability.json")["results"]
    labels = list(audit)
    conditions = list(next(iter(audit.values())))
    plt.figure(figsize=(10, 5.6))
    x = np.arange(len(labels))
    width = 0.15
    for index, condition in enumerate(conditions):
        values = [audit[name][condition]["summary"]["average_return"] for name in labels]
        plt.bar(x + (index - 2) * width, values, width, label=condition)
    plt.xticks(x, labels)
    plt.ylabel("average return")
    plt.title("Reliability matrix", fontsize=18, weight="bold", color=NAVY)
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    save(57, "reliability-matrix.png")
    clean = {name: rows["clean"] for name, rows in audit.items()}
    returns(57, "clean-return.png", clean)
    conclusion(
        57,
        "Dynamics shift and delay changed methods differently.",
        "Deployment claims require condition-wise tails, not one clean mean.",
    )


if __name__ == "__main__":
    main()

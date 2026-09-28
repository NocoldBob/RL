"""Render the four publication figures used by lessons 40 through 48."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs" / "experiments"
ASSETS = ROOT / "docs" / "assets"
NAVY = "#172554"
BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#16a34a"
RED = "#dc2626"

LESSONS = {
    40: ("Offline Dataset", "Turn interaction logs into versioned training data"),
    41: ("Behavior Cloning", "A strong expert can be easier to imitate than mixed behavior"),
    42: ("Offline DQN", "Bootstrapping can amplify unsupported actions"),
    43: ("Conservative Q-Learning", "Penalize optimistic values outside the data"),
    44: ("Implicit Q-Learning", "Learn values without maximizing over unseen actions"),
    45: ("Dataset Quality", "Coverage, quality, and mixture all change the answer"),
    46: ("Off-Policy Evaluation", "An estimate is evidence, not ground truth"),
    47: ("Offline to Online", "Fine-tune without immediately forgetting the dataset"),
    48: ("Reliability Audit", "Unseen seeds and perturbations close the loop"),
}


def load(number: int, slug: str) -> dict:
    return json.loads((EXPERIMENTS / f"{number}-{slug}.json").read_text(encoding="utf-8"))


def save_figure(number: int, name: str) -> None:
    folder = ASSETS / f"csdn-{number}"
    folder.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(folder / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


def text_card(number: int, filename: str, title: str, lines: list[str], color: str = BLUE) -> None:
    plt.figure(figsize=(10, 5.6))
    plt.axis("off")
    plt.text(0.05, 0.85, title, fontsize=24, weight="bold", color=NAVY)
    for index, line in enumerate(lines):
        plt.text(0.08, 0.66 - index * 0.14, line, fontsize=16, color=color if index == 0 else NAVY)
    save_figure(number, filename)


def bar(
    number: int, filename: str, title: str, labels: list[str], values: list[float], ylabel: str
) -> None:
    plt.figure(figsize=(10, 5.6))
    positions = np.arange(len(labels))
    colors = [BLUE, ORANGE, GREEN, RED] * 3
    bars = plt.bar(positions, values, color=colors[: len(labels)])
    plt.xticks(positions, labels, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title, fontsize=18, weight="bold", color=NAVY)
    plt.grid(axis="y", alpha=0.25)
    for item, value in zip(bars, values, strict=True):
        plt.text(
            item.get_x() + item.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom"
        )
    save_figure(number, filename)


def cover(number: int) -> None:
    title, subtitle = LESSONS[number]
    text_card(
        number,
        "cover.png",
        f"Lesson {number}  |  {title}",
        [subtitle, "Acrobot-v1 · reproducible offline RL"],
        BLUE,
    )


def conclusion(number: int, headline: str, detail: str) -> None:
    text_card(
        number, "conclusion.png", "What the experiment actually says", [headline, detail], GREEN
    )


def main() -> None:
    for number in LESSONS:
        cover(number)

    dataset = load(40, "offline-dataset")["manifest"]["datasets"]
    names = ["random", "medium", "expert", "mixed"]
    bar(
        40,
        "dataset-size.png",
        "Transitions per dataset",
        names,
        [dataset[n]["transitions"] for n in names],
        "transitions",
    )
    bar(
        40,
        "dataset-success.png",
        "Behavior success rate",
        names,
        [100 * dataset[n]["success_rate"] for n in names],
        "success (%)",
    )
    conclusion(
        40,
        "A dataset is part of the experiment, not a bag of arrays.",
        "Store termination, behavior probability, provenance, and episode boundaries.",
    )

    reports = {
        n: load(n, s)["results"]
        for n, s in [
            (41, "behavior-cloning"),
            (42, "offline-dqn"),
            (43, "cql"),
            (44, "iql"),
            (45, "dataset-quality"),
        ]
    }
    plans = {
        41: (
            "success-rate.png",
            ["BC expert", "BC mixed"],
            [reports[41]["bc_expert"], reports[41]["bc_mixed"]],
        ),
        42: ("success-rate.png", ["Offline DQN"], [reports[42]["offline_dqn"]]),
        43: ("success-rate.png", ["DQN", "CQL .1", "CQL 1", "CQL 5"], list(reports[43].values())),
        44: ("success-rate.png", ["BC", "CQL", "IQL"], list(reports[44].values())),
        45: (
            "success-rate.png",
            ["random", "medium", "expert", "mixed"],
            list(reports[45].values()),
        ),
    }
    messages = {
        41: (
            "Expert BC reached 100%; mixed BC reached only 40%.",
            "Imitation inherits the ambiguity and failures in its dataset.",
        ),
        42: (
            "Offline DQN reached 0% success on mixed data.",
            "A small Bellman loss does not prove safe policy improvement.",
        ),
        43: (
            "All mixed-data CQL settings reached 0% here.",
            "Conservatism controls extrapolation; it cannot repair every dataset/model mismatch.",
        ),
        44: (
            "IQL improved mixed-data success to 59%.",
            "Avoiding unseen-action maximization helped, but failures remain.",
        ),
        45: (
            "Expert-data CQL reached 99%; random-data CQL reached 0%.",
            "More transitions did not beat higher-quality behavior.",
        ),
    }
    for number, (filename, labels, rows) in plans.items():
        success = [100 * row["summary"]["success_rate"] for row in rows]
        returns = [row["summary"]["average_return"] for row in rows]
        bar(number, filename, "Hidden-seed success rate", labels, success, "success (%)")
        bar(number, "average-return.png", "Hidden-seed average return", labels, returns, "return")
        conclusion(number, *messages[number])

    ope = load(46, "off-policy-evaluation")["results"]
    labels = list(ope)
    for metric, filename in [("WIS", "wis-vs-truth.png"), ("FQE", "fqe-vs-truth.png")]:
        estimates = [
            ope[name]["importance_sampling"]["weighted_is"]
            if metric == "WIS"
            else ope[name]["fqe_estimate"]
            for name in labels
        ]
        values = estimates + [ope[name]["true_discounted_return"]["mean"] for name in labels]
        bar(
            46,
            filename,
            f"{metric} estimates and true discounted returns",
            [f"{x}-{metric}" for x in labels] + [f"{x}-true" for x in labels],
            values,
            "discounted return",
        )
    conclusion(
        46,
        "IS collapsed and WIS/FQE stayed biased.",
        "Use several estimators, support diagnostics, and uncertainty—not one magic score.",
    )

    online = load(47, "offline-to-online")["results"]
    labels = list(online)
    final_rows = [online[name]["history"]["evaluations"][-1] for name in labels]
    bar(
        47,
        "final-return.png",
        "Return after 5k online steps",
        labels,
        [r["average_return"] for r in final_rows],
        "return",
    )
    plt.figure(figsize=(10, 5.6))
    for name in labels:
        xs = [row["step"] for row in online[name]["history"]["evaluations"]]
        ys = [row["average_return"] for row in online[name]["history"]["evaluations"]]
        plt.plot(xs, ys, marker="o", label=name)
    plt.title("Offline-to-online learning curves", fontsize=18, weight="bold", color=NAVY)
    plt.xlabel("online environment steps")
    plt.ylabel("average return")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(47, "learning-curves.png")
    conclusion(
        47,
        "Mixed replay recovered fastest in this 5k-step budget.",
        "Direct and conservative updates showed that initialization alone is insufficient.",
    )

    audit = load(48, "offline-reliability")["results"]
    labels = list(audit)
    conditions = ["clean", "observation_noise", "sticky_actions"]
    plt.figure(figsize=(10, 5.6))
    width = 0.24
    x = np.arange(len(labels))
    for index, condition in enumerate(conditions):
        values = [
            100 * audit[name]["conditions"][condition]["summary"]["success_rate"] for name in labels
        ]
        plt.bar(x + (index - 1) * width, values, width, label=condition)
    plt.xticks(x, labels)
    plt.ylabel("success (%)")
    plt.title("Reliability matrix", fontsize=18, weight="bold", color=NAVY)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure(48, "reliability-matrix.png")
    bar(
        48,
        "clean-success.png",
        "Unseen-seed clean success",
        labels,
        [100 * audit[name]["conditions"]["clean"]["summary"]["success_rate"] for name in labels],
        "success (%)",
    )
    conclusion(
        48,
        "Observation noise changed the ranking and exposed brittle policies.",
        "Report tails and perturbations before calling an offline policy deployable.",
    )


if __name__ == "__main__":
    main()

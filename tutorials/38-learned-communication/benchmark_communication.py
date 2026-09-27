"""Ablate the learned message channel at evaluation time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppo_agents import MultiAgentPPO, PPOConfig
from spread_env import SpreadConfig, make_env, run_episode, summarize
from value_agents import padded_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/marl-ppo/communication/checkpoints/best.pt")
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/38-learned-communication.json")
    )
    args = parser.parse_args()
    agent = MultiAgentPPO(PPOConfig(mode="communication"))
    step = agent.load(args.checkpoint)
    results = {}
    for label, dropout in (("messages", 0.0), ("half_dropped", 0.5), ("no_messages", 1.0)):
        environment = make_env(SpreadConfig())

        def policy(observations, env, dropout_probability=dropout):
            prepared = agent.prepare(observations)
            actions, _, _ = agent.act(
                prepared,
                padded_state(env.state()),
                deterministic=True,
                message_dropout=dropout_probability,
            )
            return {f"agent_{index}": int(action) for index, action in enumerate(actions)}

        try:
            rows = [
                run_episode(environment, policy, seed=70_000 + index)
                for index in range(args.episodes)
            ]
        finally:
            environment.close()
        results[label] = {"message_dropout": dropout, "summary": summarize(rows), "episodes": rows}
    report = {"experiment": "38-learned-communication", "checkpoint_step": step, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: row["summary"] for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()

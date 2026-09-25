"""Render one LunarLander continuous baseline episode."""

from __future__ import annotations

import argparse

from lunar_baselines import POLICY_NAMES, run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a LunarLander baseline")
    parser.add_argument("policy", choices=POLICY_NAMES, nargs="?", default="heuristic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result, _ = run_episode(args.policy, args.seed, render_mode="human")
    print(result.payload())


if __name__ == "__main__":
    main()

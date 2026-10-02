from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ken43 import (
    AlwaysBlockBot,
    AlwaysDIBot,
    AlwaysJinraiBot,
    AlwaysMashBot,
    AlwaysODDPBot,
    AlwaysThrowBot,
    BackdashHeavyBot,
    Ken43ParallelEnv,
    MixedBot,
    RandomBot,
    select_bot_action,
    update_bot,
)


POLICIES = {
    "always_block": AlwaysBlockBot,
    "always_throw": AlwaysThrowBot,
    "always_mash": AlwaysMashBot,
    "always_od_dp": AlwaysODDPBot,
    "always_di": AlwaysDIBot,
    "always_jinrai": AlwaysJinraiBot,
    "backdash_heavy": BackdashHeavyBot,
    "random": RandomBot,
    "mixed": MixedBot,
}


def play(row_type, column_type, episodes: int, seed: int):
    returns: list[float] = []
    reasons: Counter[str] = Counter()
    for episode in range(episodes):
        env = Ken43ParallelEnv(initial_advantages=(26, 38, 43))
        observations, _ = env.reset(seed=seed + episode)
        bots = [row_type(), column_type()]
        total = 0.0
        while env.agents:
            actions = {
                agent: select_bot_action(
                    bots[actor], observations[agent], env.command_index,
                    env._rng, env, actor,
                )
                for actor, agent in enumerate(env.possible_agents)
            }
            observations, rewards, terminated, truncated, infos = env.step(actions)
            for actor, bot in enumerate(bots):
                update_bot(bot, infos, env, actor)
            total += rewards["player_0"]
            if all(terminated.values()) or all(truncated.values()):
                reasons[infos["player_0"]["terminal_reason"] or "unknown"] += 1
                break
        returns.append(total)
    return float(np.mean(returns)), reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ken43 scripted-policy payoff matrix")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--dominance-margin", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Ken43/results/pairwise_payoffs.json"),
    )
    args = parser.parse_args()
    names = list(POLICIES)
    matrix: dict[str, dict[str, float]] = {name: {} for name in names}
    terminal_reasons: dict[str, dict[str, dict[str, int]]] = {name: {} for name in names}
    for row_index, row in enumerate(names):
        for column_index, column in enumerate(names):
            value, reasons = play(
                POLICIES[row],
                POLICIES[column],
                args.episodes,
                args.seed + 10000 * row_index + 100 * column_index,
            )
            matrix[row][column] = value
            terminal_reasons[row][column] = dict(reasons)

    role_balanced: dict[str, dict[str, float]] = {name: {} for name in names}
    for row in names:
        for column in names:
            role_balanced[row][column] = 0.5 * (matrix[row][column] - matrix[column][row])
    dominant = [
        row
        for row in names
        if all(
            row == column or role_balanced[row][column] > args.dominance_margin
            for column in names
        )
    ]
    result = {
        "episodes_per_ordered_pair": args.episodes,
        "policies": names,
        "attacker_role_payoff": matrix,
        "role_balanced_payoff": role_balanced,
        "terminal_reasons": terminal_reasons,
        "dominance_margin": args.dominance_margin,
        "universal_dominant_policies": dominant,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print("policy," + ",".join(names))
    for row in names:
        print(row + "," + ",".join(f"{matrix[row][column]:.2f}" for column in names))
    print("universal_dominant_policies=" + (",".join(dominant) if dominant else "none"))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

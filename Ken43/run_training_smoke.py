from __future__ import annotations

import argparse

import numpy as np

from ken43 import Ken43ParallelEnv, RandomBot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run masked random Ken43 rollouts")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = Ken43ParallelEnv(max_frames=args.max_frames)
    returns = {agent: [] for agent in env.possible_agents}
    reasons: dict[str, int] = {}

    for episode in range(args.episodes):
        observations, _ = env.reset(seed=args.seed + episode)
        totals = {agent: 0.0 for agent in env.possible_agents}
        bots = {agent: RandomBot() for agent in env.possible_agents}
        reason = "max_frames"
        while env.agents:
            actions = {}
            for agent in env.agents:
                actions[agent] = bots[agent].act(
                    observations[agent], env.command_index, rng
                )
            observations, rewards, terminated, truncated, infos = env.step(actions)
            for agent, reward in rewards.items():
                totals[agent] += reward
            if all(terminated.values()) or all(truncated.values()):
                reason = infos["player_0"].get("terminal_reason") or "max_frames"
        for agent in env.possible_agents:
            returns[agent].append(totals[agent])
        reasons[reason] = reasons.get(reason, 0) + 1

    print(f"episodes={args.episodes} action_count={env.action_space('player_0').n}")
    for agent in env.possible_agents:
        print(f"{agent}_mean_return={np.mean(returns[agent]):.2f}")
    print("terminal_reasons=" + ", ".join(f"{key}:{value}" for key, value in sorted(reasons.items())))


if __name__ == "__main__":
    main()

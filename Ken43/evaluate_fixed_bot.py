from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ken43 import AlwaysBlockBot, Ken43GymEnv, MashBot, MixedBot, ReversalBot, ThrowBot


BOT_TYPES = {
    "block": AlwaysBlockBot,
    "mash": MashBot,
    "throw": ThrowBot,
    "reversal": ReversalBot,
    "mixed": MixedBot,
}


def info_label(action: np.ndarray, commands: tuple[str, ...]) -> str:
    return commands[int(np.asarray(action).item())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Ken43 MaskablePPO checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bot", choices=sorted(BOT_TYPES), default="block")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--drive-reward-weight", type=float, default=1.0)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(args.seed)

    env = Ken43GymEnv(
        opponent=BOT_TYPES[args.bot](),
        max_frames=600,
        drive_reward_weight=args.drive_reward_weight,
    )
    model = MaskablePPO.load(args.checkpoint)
    returns: list[float] = []
    lengths: list[int] = []
    final_drive: list[tuple[float, float]] = []
    drive_returns: list[float] = []
    terminal_reasons: Counter[str] = Counter()
    commands: Counter[str] = Counter()
    returns_by_reason: dict[str, list[float]] = defaultdict(list)
    timeout_penalty_returns: list[float] = []
    time_pressure_returns: list[float] = []

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        episode_return = 0.0
        drive_return = 0.0
        timeout_penalty_return = 0.0
        time_pressure_return = 0.0
        length = 0
        while True:
            action, _ = model.predict(
                observation,
                deterministic=not args.stochastic,
                action_masks=env.action_masks(),
            )
            action_value = int(np.asarray(action).item())
            commands[info_label(action_value, env.commands)] += 1
            observation, reward, terminated, truncated, info = env.step(action_value)
            episode_return += float(reward)
            drive_return += float(info["reward_components"]["drive"])
            drive_return += float(info["reward_components"]["event_bonus"])
            timeout_penalty_return += float(
                info["reward_components"]["timeout_penalty"]
            )
            time_pressure_return += float(
                info["reward_components"]["time_pressure"]
            )
            length += 1
            if terminated or truncated:
                returns.append(episode_return)
                drive_returns.append(drive_return)
                lengths.append(length)
                players = env.parallel.core.combat_state.players
                final_drive.append((players[0].drive, players[1].drive))
                reason = info.get("terminal_reason") or ("max_frames" if truncated else "unknown")
                terminal_reasons[str(reason)] += 1
                returns_by_reason[str(reason)].append(episode_return)
                timeout_penalty_returns.append(timeout_penalty_return)
                time_pressure_returns.append(time_pressure_return)
                break

    return_array = np.asarray(returns, dtype=np.float64)
    drive_array = np.asarray(final_drive, dtype=np.float64)
    print(f"checkpoint={args.checkpoint}")
    print(
        f"bot={args.bot} episodes={args.episodes} "
        f"policy={'stochastic' if args.stochastic else 'deterministic'}"
    )
    print(
        "return "
        f"mean={return_array.mean():.2f} std={return_array.std():.2f} "
        f"min={return_array.min():.2f} max={return_array.max():.2f}"
    )
    print(f"episode_length_mean={np.mean(lengths):.2f}")
    print(f"drive_and_event_reward_mean={np.mean(drive_returns):.2f}")
    print(f"timeout_penalty_mean={np.mean(timeout_penalty_returns):.3f}")
    print(f"time_pressure_mean={np.mean(time_pressure_returns):.3f}")
    print(f"final_drive_mean player={drive_array[:, 0].mean():.2f} opponent={drive_array[:, 1].mean():.2f}")
    print("terminal_reasons=" + ",".join(f"{key}:{value}" for key, value in terminal_reasons.items()))
    print(
        "return_by_terminal="
        + ",".join(
            f"{key}:{np.mean(values):.2f}" for key, values in sorted(returns_by_reason.items())
        )
    )
    print("top_commands=" + ",".join(f"{key}:{value}" for key, value in commands.most_common(10)))


if __name__ == "__main__":
    main()

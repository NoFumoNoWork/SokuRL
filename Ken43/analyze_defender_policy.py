from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ken43 import HeuristicOffenseBot, Ken43GymEnv


DEFENSIVE_SUCCESSES = {
    "pressure_escape",
    "throw_tech",
    "side_switch",
    "defender_disengage",
    "corner_escape",
    "counter_di_escape",
    "player_0_drive_break",
}
DEFENSIVE_FAILURES = {
    "defender_thrown",
    "di_wall_splat",
    "player_1_drive_break",
}


def summarize_counter(counter: Counter, limit: int = 20) -> dict[str, int]:
    return dict(counter.most_common(limit))


def evaluate(label, episodes, seed, policy=None, deterministic=True):
    offense = HeuristicOffenseBot()
    env = Ken43GymEnv(
        opponent=offense,
        learning_agent="player_1",
        max_frames=600,
    )
    terminals = Counter()
    submitted = Counter()
    starts = Counter()
    defender_contacts = Counter()
    attacker_contacts = Counter()
    offense_macros = Counter()
    returns: list[float] = []
    lengths: list[int] = []
    final_drive = [[], []]
    returns_by_reason: dict[str, list[float]] = defaultdict(list)
    reward_components: Counter[str] = Counter()

    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total = 0.0
        length = 0
        while True:
            mask = env.action_masks()
            if policy == "block":
                action = env.command_index["guard:crouching"]
            elif policy == "random":
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                action, _ = policy.predict(
                    observation,
                    deterministic=deterministic,
                    action_masks=mask,
                )
                action = int(np.asarray(action).item())
            observation, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            length += 1
            submitted[str(info["submitted_command"])] += 1
            for name, value in info["reward_components"].items():
                reward_components[name] += float(value)
            for event in info["events"]:
                event_type = str(event["type"])
                actor = event.get("actor")
                move = str(event.get("data", {}).get("action", "unknown"))
                if event_type in {"ActionStarted", "ReversalStarted", "BranchStarted"}:
                    if actor == 1:
                        starts[move] += 1
                    elif actor == 0 and offense.macro is not None:
                        offense_macros[offense.macro] += 1
                if event_type in {
                    "Hit", "Block", "Whiff", "Throw", "ThrowWhiff", "ThrowClash",
                    "ArmorAbsorb", "ArmorBreak", "CounterDI",
                }:
                    destination = defender_contacts if actor == 1 else attacker_contacts
                    destination[f"{move}:{event_type}"] += 1
            if terminated or truncated:
                reason = str(info.get("terminal_reason") or "unknown")
                terminals[reason] += 1
                returns.append(total)
                lengths.append(length)
                returns_by_reason[reason].append(total)
                final_drive[0].append(env.parallel.core.combat_state.players[0].drive)
                final_drive[1].append(env.parallel.core.combat_state.players[1].drive)
                break

    successes = sum(terminals[reason] for reason in DEFENSIVE_SUCCESSES)
    failures = sum(terminals[reason] for reason in DEFENSIVE_FAILURES)
    timeouts = terminals["max_frames"]
    result = {
        "label": label,
        "episodes": episodes,
        "deterministic": deterministic,
        "mean_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "defensive_success_rate": successes / episodes,
        "defensive_failure_rate": failures / episodes,
        "timeout_rate": timeouts / episodes,
        "terminal_reasons": summarize_counter(terminals),
        "mean_return_by_reason": {
            reason: float(np.mean(values)) for reason, values in returns_by_reason.items()
        },
        "mean_final_drive": {
            "attacker": float(np.mean(final_drive[0])),
            "defender": float(np.mean(final_drive[1])),
        },
        "reward_component_totals": dict(reward_components),
        "submitted_commands": summarize_counter(submitted, 30),
        "defender_action_starts": summarize_counter(starts, 30),
        "defender_contacts": summarize_counter(defender_contacts, 40),
        "attacker_contacts": summarize_counter(attacker_contacts, 40),
        "offense_macro_action_starts": summarize_counter(offense_macros),
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="*", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=24000)
    parser.add_argument("--stochastic-final", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Ken43/results/defender_vs_heuristic_analysis.json"),
    )
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(args.seed)
    results = [
        evaluate("always_block", args.episodes, args.seed, policy="block"),
        evaluate("random_legal", args.episodes, args.seed, policy="random"),
    ]
    for checkpoint in args.checkpoints:
        model = MaskablePPO.load(checkpoint, device="cuda")
        results.append(evaluate(checkpoint.stem, args.episodes, args.seed, policy=model))
    if args.stochastic_final and args.checkpoints:
        checkpoint = args.checkpoints[-1]
        model = MaskablePPO.load(checkpoint, device="cuda")
        results.append(evaluate(
            checkpoint.stem + "_stochastic",
            args.episodes,
            args.seed,
            policy=model,
            deterministic=False,
        ))

    payload = {"episodes_per_policy": args.episodes, "seed": args.seed, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print("label,return,length,success,failure,timeout")
    for result in results:
        print(
            f"{result['label']},{result['mean_return']:.3f},{result['mean_length']:.1f},"
            f"{result['defensive_success_rate']:.1%},"
            f"{result['defensive_failure_rate']:.1%},{result['timeout_rate']:.1%}"
        )
        print("  terminals=" + json.dumps(result["terminal_reasons"], sort_keys=True))
        print("  starts=" + json.dumps(result["defender_action_starts"], sort_keys=True))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

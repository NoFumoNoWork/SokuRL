from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ken43 import Ken43GymEnv, MixedBot


def format_counter(counter: Counter, limit: int = 20) -> str:
    return ",".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=18000)
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(args.seed)
    env = Ken43GymEnv(opponent=MixedBot(), max_frames=600)
    model = MaskablePPO.load(args.checkpoint)
    starts = [Counter(), Counter()]
    contacts = [Counter(), Counter()]
    first_defenses = Counter()
    terminal_reasons = Counter()
    defender_started = False

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        defender_started = False
        while True:
            action, _ = model.predict(
                observation,
                deterministic=False,
                action_masks=env.action_masks(),
            )
            observation, _, terminated, truncated, info = env.step(
                int(np.asarray(action).item())
            )
            for event in info["events"]:
                event_type = event["type"]
                actor = event.get("actor")
                data = event.get("data", {})
                move = str(data.get("action", "unknown"))
                if event_type in {"ActionStarted", "ReversalStarted", "BranchStarted"}:
                    if actor is not None:
                        starts[int(actor)][move] += 1
                        if int(actor) == 1 and not defender_started:
                            first_defenses[move] += 1
                            defender_started = True
                if event_type in {
                    "Hit", "Block", "Whiff", "Throw", "ThrowWhiff", "ThrowClash",
                    "ArmorAbsorb", "ArmorBreak", "CounterDI",
                } and actor is not None:
                    contacts[int(actor)][f"{move}:{event_type}"] += 1
            if terminated or truncated:
                terminal_reasons[str(info.get("terminal_reason", "unknown"))] += 1
                if not defender_started:
                    first_defenses["guard_or_no_action"] += 1
                break

    print(f"episodes={args.episodes}")
    print("terminal_reasons=" + format_counter(terminal_reasons))
    print("attacker_action_starts=" + format_counter(starts[0], 30))
    print("attacker_contacts=" + format_counter(contacts[0], 40))
    print("defender_first_response=" + format_counter(first_defenses, 20))
    print("defender_action_starts=" + format_counter(starts[1], 30))
    print("defender_contacts=" + format_counter(contacts[1], 40))


if __name__ == "__main__":
    main()

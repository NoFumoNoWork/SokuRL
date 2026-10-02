from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ken43 import (
    AlwaysBlockBot, AlwaysDIBot, AlwaysJinraiBot, AlwaysMashBot, AlwaysODDPBot,
    AlwaysThrowBot, BackdashHeavyBot, Ken43ParallelEnv, MixedBot, RandomBot,
    select_bot_action, update_bot,
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


def semantic_valid_count(env: Ken43ParallelEnv, actor: int) -> int:
    return int(np.sum(env._mask(actor)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument(
        "--output", type=Path, default=Path("Ken43/results/v1_diagnostics_600f.json")
    )
    parser.add_argument(
        "--episode-log",
        type=Path,
        default=Path("Ken43/results/v1_episode_diagnostics_600f.jsonl.gz"),
    )
    args = parser.parse_args()

    reasons: Counter[str] = Counter()
    commands: Counter[str] = Counter()
    max_frame_states: Counter[str] = Counter()
    timeout_classification: Counter[str] = Counter()
    max_frame_phase_slots: Counter[str] = Counter()
    max_frame_episodes_with_phase: Counter[str] = Counter()
    od_dp: Counter[str] = Counter()
    di: Counter[str] = Counter()
    jinrai: Counter[str] = Counter()
    frame_kill: Counter[str] = Counter()
    schedule_delays: Counter[str] = Counter()
    drive_sources = [Counter(), Counter()]
    episode_logs: list[dict] = []
    lengths: list[int] = []
    returns: list[float] = []
    sample = Ken43ParallelEnv(initial_advantages=(26, 38, 43))
    observations, _ = sample.reset(seed=args.seed)
    initial_valid = {agent: semantic_valid_count(sample, i) for i, agent in enumerate(sample.possible_agents)}
    sample.core.combat_state.frame = 42
    wakeup_valid = {agent: semantic_valid_count(sample, i) for i, agent in enumerate(sample.possible_agents)}

    episode_id = 0
    for left_name, left_type in POLICIES.items():
        for right_name, right_type in POLICIES.items():
            for _ in range(args.episodes):
                env = Ken43ParallelEnv(initial_advantages=(26, 38, 43))
                observations, _ = env.reset(seed=args.seed + episode_id)
                bots = [left_type(), right_type()]
                total = 0.0
                length = 0
                while env.agents:
                    actions = {
                        agent: select_bot_action(
                            bots[i], observations[agent], env.command_index,
                            env._rng, env, i,
                        )
                        for i, agent in enumerate(env.possible_agents)
                    }
                    observations, rewards, terminated, truncated, infos = env.step(actions)
                    for i, bot in enumerate(bots):
                        update_bot(bot, infos, env, i)
                    total += rewards["player_0"]
                    length += 1
                    commands.update(info["submitted_command"] for info in infos.values())
                    if all(terminated.values()) or all(truncated.values()):
                        reason = infos["player_0"]["terminal_reason"] or "unknown"
                        reasons[reason] += 1
                        diagnostics = infos["player_0"]["episode_diagnostics"]
                        episode_logs.append({
                            "episode": episode_id,
                            "player_0_policy": left_name,
                            "player_1_policy": right_name,
                            **diagnostics,
                        })
                        od_dp.update(diagnostics["od_dp"])
                        di.update(diagnostics["di"])
                        jinrai.update(diagnostics["jinrai"])
                        frame_kill.update(diagnostics["frame_kill"])
                        schedule_delays.update(diagnostics["schedule_delay_histogram"])
                        for actor in range(2):
                            drive_sources[actor].update(diagnostics["drive_change_sources"][actor])
                        if reason == "max_frames":
                            activity = diagnostics["activity_at_end"]
                            for key, value in activity.items():
                                max_frame_states[key] += int(value)
                            phases = [
                                player["phase"]
                                for player in diagnostics["last_trace"][-1]["players"]
                            ]
                            max_frame_phase_slots.update(phases)
                            max_frame_episodes_with_phase.update(set(phases))
                            timeout_classification[
                                diagnostics["timeout_analysis"]["classification"]
                            ] += 1
                        break
                returns.append(total)
                lengths.append(length)
                episode_id += 1

    total_commands = sum(commands.values())
    completed_od = float(od_dp.get("completed_drive_samples", 0.0))
    od_dp["mean_net_drive_completed"] = (
        float(od_dp.get("net_drive_total_completed", 0.0)) / completed_od
        if completed_od else 0.0
    )
    result = {
        "episodes": len(lengths),
        "observation_shape": [32],
        "action_count": sample.action_space("player_0").n,
        "action_parameterization": ["mode", "move", "delay"],
        "initial_semantic_valid_actions": initial_valid,
        "wakeup_semantic_valid_actions": wakeup_valid,
        "mean_episode_length": float(np.mean(lengths)),
        "mean_player_0_return": float(np.mean(returns)),
        "terminal_reasons": dict(reasons),
        "max_frames_ratio": reasons["max_frames"] / max(1, len(lengths)),
        "max_frame_states": dict(max_frame_states),
        "timeout_classification": dict(timeout_classification),
        "max_frame_phase_slots": dict(max_frame_phase_slots),
        "max_frame_episodes_with_phase": dict(max_frame_episodes_with_phase),
        "od_dp_outcomes": dict(od_dp),
        "di_outcomes": dict(di),
        "jinrai_breakdown": dict(jinrai),
        "frame_kill_breakdown": dict(frame_kill),
        "schedule_delay_histogram": dict(schedule_delays),
        "drive_change_sources": [dict(item) for item in drive_sources],
        "usage": {
            "od_dp": sum(value for key, value in commands.items() if "shoryuken_OD" in key),
            "jinrai": sum(value for key, value in commands.items() if "jinrai" in key or "kazekama" in key or "gorai" in key or "senka" in key),
            "frame_kill": sum(value for key, value in commands.items() if any(move in key for move in ("5MK", "5HK", "2HK"))),
            "noop": sum(value for key, value in commands.items() if "noop" in key),
            "total_commands": total_commands,
        },
        "top_commands": commands.most_common(15),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    args.episode_log.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        json.dumps(item, separators=(",", ":")) + "\n" for item in episode_logs
    )
    if args.episode_log.suffix == ".gz":
        with gzip.open(args.episode_log, "wt", encoding="ascii") as handle:
            handle.write(serialized)
    else:
        args.episode_log.write_text(serialized, encoding="ascii")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Run state-based decision trees in both player slots against each other."""
from __future__ import annotations

from collections import Counter
import ctypes
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from soku_rl.baselines import Fighter, Observation, Projectile, TreeConfig, TreePolicy
from bridge_shared import BridgeClient, FRAME_RING_CAPACITY
from headless_validation import wait_for_frame_zero
from interaction_benchmark import wait_group
from unlimited_benchmark import FRAME_SIZE, _drain_fast
import sokurl


ROOT = Path(__file__).resolve().parents[1]


def observe(state, player_index):
    if player_index not in (0, 1):
        raise ValueError("player_index must be zero or one")
    if state.p1ObjectOverflow or state.p2ObjectOverflow:
        raise RuntimeError("object observation overflow; this episode is invalid")
    players = (state.p1, state.p2)

    def fighter(raw):
        if raw.maxSpirit == 0:
            raise RuntimeError("player maxSpirit is zero")
        return Fighter(raw.x, raw.y, raw.hp, raw.spirit / raw.maxSpirit,
                       raw.actionId, bool(raw.airborne), raw.hitstop)

    objects, count = ((state.p2Objects, state.p2ObjectCount) if player_index == 0
                      else (state.p1Objects, state.p1ObjectCount))
    projectiles = tuple(Projectile(obj.x, obj.y, obj.speedX, obj.speedY)
                        for obj in objects[:count] if obj.isActive and obj.hitBoxCount)
    return Observation(state.frameId, fighter(players[player_index]),
                       fighter(players[1 - player_index]), projectiles)


def play_batch(config, seed, pairs):
    processes, clients, buffers = [], [], []
    try:
        started = time.perf_counter()
        processes = sokurl._launch_vs_group_from_title(
            len(pairs), 180.0, headless=True, unlimited=True,
            seed=seed, pause_at_start=True,
        )
        states, policies, rules, records = [], [], [], []
        for process, pair in zip(processes, pairs, strict=True):
            client = BridgeClient(process.pid)
            clients.append(client)
            state = wait_for_frame_zero(client, process.pid)
            states.append(state)
            buffers.append((ctypes.c_ubyte * (FRAME_RING_CAPACITY * FRAME_SIZE))())
            policy_pair = []
            for name in pair:
                values = dict(config["tree"])
                values.update(config["overrides"][name])
                policy_pair.append(TreePolicy(TreeConfig(style=name, **values)))
            policies.append(policy_pair)
            rules.append([Counter(), Counter()])
            records.append({
                "seed": seed, "p1_policy": pair[0], "p2_policy": pair[1],
                "p1_character": state.p1.characterId, "p2_character": state.p2.characterId,
                "initial_hp": [state.p1.hp, state.p2.hp],
                "max_objects": [0, 0], "minimum_hp": [state.p1.hp, state.p2.hp],
            })
        launch_seconds = time.perf_counter() - started
        sampling_started = time.perf_counter()
        active = list(range(len(pairs)))
        policy_seconds = 0.0
        simulation_steps = 0
        for frame in range(1, config["max_frames"] + 1):
            decisions = []
            policy_started = time.perf_counter()
            for index in active:
                pair = [policy.act(observe(states[index], side))
                        for side, policy in enumerate(policies[index])]
                for side, decision in enumerate(pair):
                    rules[index][side][decision.rule] += 1
                decisions.append(pair)
            policy_seconds += time.perf_counter() - policy_started
            active_clients = [clients[index] for index in active]
            sequences = [client.step_with_inputs(pair[0].inputs, pair[1].inputs)
                         for client, pair in zip(active_clients, decisions, strict=True)]
            snapshots = wait_group(active_clients, sequences, frame, "ready")
            finished = []
            for index, snapshot in zip(active, snapshots, strict=True):
                state = snapshot.latest
                if state.p1ObjectOverflow or state.p2ObjectOverflow:
                    raise RuntimeError("object observation overflow")
                states[index] = state
                record = records[index]
                for side, (hp, count) in enumerate(((state.p1.hp, state.p1ObjectCount),
                                                   (state.p2.hp, state.p2ObjectCount))):
                    record["minimum_hp"][side] = min(record["minimum_hp"][side], hp)
                    record["max_objects"][side] = max(record["max_objects"][side], count)
                _drain_fast(clients[index], buffers[index])
                knocked_out = state.p1.hp <= 0 or state.p2.hp <= 0
                if knocked_out or frame == config["max_frames"]:
                    winner = "timeout"
                    if knocked_out:
                        winner = "draw" if state.p1.hp <= 0 and state.p2.hp <= 0 else (
                            "p1" if state.p2.hp <= 0 else "p2")
                    record.update({
                        "frames": frame, "outcome": winner,
                        "final_hp": [state.p1.hp, state.p2.hp],
                        "final_hash": f"{state.stateHash:016X}",
                        "dropped_frames": snapshot.dropped_frames,
                        "rules": [dict(counter) for counter in rules[index]],
                    })
                    if snapshot.dropped_frames:
                        raise RuntimeError("frame recording was incomplete")
                    finished.append(index)
                    print(f"seed={seed} {pairs[index]}: {winner}, frame={frame}, "
                          f"hp={record['final_hp']}", flush=True)
            simulation_steps += len(active)
            active = [index for index in active if index not in finished]
            if not active:
                break
        sampling_seconds = time.perf_counter() - sampling_started
        return {
            "seed": seed, "launch_seconds": launch_seconds,
            "sampling_seconds": sampling_seconds, "policy_seconds": policy_seconds,
            "simulation_steps": simulation_steps,
            "steps_per_second_including_policy": simulation_steps / sampling_seconds,
            "games": records,
        }
    finally:
        for client in clients:
            client.close()
        for process in processes:
            if process.is_running():
                sokurl.shutdown(5.0, process.pid)


@hydra.main(version_base="1.3", config_path="../config", config_name="baselines")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    if config["max_frames"] < 1 or not config["seeds"]:
        raise ValueError("positive max_frames and nonempty seeds are required")
    if len(config["profiles"]) < 2 or len(set(config["profiles"])) != len(config["profiles"]):
        raise ValueError("at least two distinct profiles are required")
    for name in config["profiles"]:
        TreeConfig(style=name, **(config["tree"] | config["overrides"][name]))
    sokurl._validate_game()
    destination = ROOT / config["output"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {"success": False, "config": config, "batches": [],
              "termination": "first knockout or frame limit; timeout is not a win"}
    pairs = list(itertools.permutations(config["profiles"], 2))
    try:
        for seed in config["seeds"]:
            batch = play_batch(config, seed, pairs)
            report["batches"].append(batch)
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["success"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report={destination}", flush=True)


if __name__ == "__main__":
    main()

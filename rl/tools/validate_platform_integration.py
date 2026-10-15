"""Validate imported worker/state/vector boundaries against unchanged ABI12."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import os

import numpy as np
import psutil

from bridge_shared import BridgeClient
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.env.vector_env import TwoPlayerVectorEnv
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.encoding import AGENTS, decode_action, encode_action


VISIBILITY = dict(pixel_quantum=8, minimum_alpha=.5, hp_quantum=.02,
                  spirit_quantum=.1, player_width=48., player_height=96.,
                  object_diameter=16., minimum_visible_fraction=.25)


def run(workers, frames, output):
    root = Path(__file__).resolve().parents[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "workers": workers, "frames": frames,
              "long_training": "NOT STARTED", "checks": {}}
    def save():
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    if psutil.virtual_memory().available < workers * 550_000_000 + 1_000_000_000:
        report.update(status="BLOCKED", error="insufficient available RAM for bounded instance test")
        save()
        return 2
    config = EpisodeConfig(frames + 100, 4, 1, 0, "state", VisibilityConfig(**VISIBILITY), LEGACY_MATCH)
    backend = WorkerBackend([sys.executable, "-B", str(root / "tools/rollout_worker.py")],
                            str(root), str(output.with_suffix(".worker.log")), 120, 40,
                            True, os.environ['SOKURL_GAME_DIR'], fresh_reset=True,
                            cards_disabled=True, fixed_weather="clear")
    backend.configure_observation(config.backend_observation())
    env = TwoPlayerVectorEnv(backend, workers, config)
    clients = {}
    def reset(seeds):
        started = time.monotonic()
        obs, infos = env.reset(seeds)
        return obs, infos, time.monotonic() - started
    def advance():
        trajectory = {slot: [] for slot in range(workers)}
        for frame in range(frames):
            inputs = ((0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0, 0)) if frame < 8 else (
                ((-1 if frame % 80 < 40 else 1), 0, int(frame % 30 < 3), 0, int(frame % 90 < 3), 0, 0, 0),
                ((1 if frame % 80 < 40 else -1), 0, int(frame % 40 < 3), int(frame % 100 < 3), 0, 0, 0, 0))
            actions = {slot: {agent: encode_action(inputs[seat]) for seat, agent in enumerate(AGENTS)}
                       for slot in range(workers)}
            obs, _, terms, truncs, infos = env.step(actions)
            for slot in range(workers):
                assert not any(terms[slot].values()) and not any(truncs[slot].values())
                snapshot = clients[slot].snapshot()
                assert snapshot.latest.activeWeather == 21
                trajectory[slot].append((snapshot.latest.stateHash,
                    tuple(hashlib.sha256(obs[slot][agent].tobytes()).hexdigest() for agent in AGENTS)))
        return trajectory
    try:
        report["worker"] = {key: backend.identity[key] for key in (
            "protocol", "bridge_abi", "fresh_reset", "record_replays", "cards_disabled", "fixed_weather")}
        seeds = {slot: 1732 + slot for slot in range(workers)}
        first, infos, cold_seconds = reset(seeds)
        statuses = backend.request("status", list(seeds))
        for slot in range(workers):
            pid = statuses[slot]["pid"]
            clients[slot] = BridgeClient(pid)
            assert clients[slot].fresh_reset_enabled
        pids = {slot: clients[slot].pid for slot in clients}
        reference = advance()
        repeated, _, warm_seconds = reset(seeds)
        assert all(np.array_equal(first[s][agent], repeated[s][agent]) for s in seeds for agent in AGENTS)
        actual = advance()
        assert reference == actual, "same-seed vector/raw/observation trajectory differs"
        report["checks"]["same_seed_native_and_public_trajectory"] = "PASS"
        previous = {slot: (clients[slot].snapshot().game_frame,
                          clients[slot].snapshot().latest.stateHash) for slot in range(1, workers)}
        _, _, partial_seconds = reset({0: 7007})
        for slot, expected in previous.items():
            snap = clients[slot].snapshot()
            assert (snap.game_frame, snap.latest.stateHash) == expected
        assert all(clients[slot].pid == pids[slot] for slot in clients)
        report["checks"]["partial_new_seed_reset_isolation_same_pid"] = "PASS"
        report["checks"]["new_episode_history_and_delay_queues"] = "PASS"
        report.update(status="PASS", pids=pids, cold_launch_seconds=cold_seconds,
                      warm_reset_seconds=warm_seconds, partial_reset_seconds=partial_seconds)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        for client in clients.values():
            client.close()
        env.close()
        save()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 2 or not 8 <= args.frames <= 2000:
        parser.error("at least two workers and 8..2000 frames are required")
    raise SystemExit(run(args.workers, args.frames, args.output))

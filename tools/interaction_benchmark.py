"""Measure synchronized two-player sampling and process restart costs."""
from __future__ import annotations

import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

from bridge_shared import BridgeClient, FRAME_RING_CAPACITY, wait_for_steps
from headless_validation import fixed_trace, wait_for_frame_zero
from unlimited_benchmark import FRAME_SIZE, _drain_fast
import sokurl

RUN_STATE_PAUSED = 1


def wait_group(clients, sequences, frame, polling):
    if polling == "ready":
        return wait_for_steps(clients, sequences, [frame] * len(clients), 10.0)
    deadline = time.perf_counter() + 10.0
    pending = set(range(len(clients)))
    snapshots = [None] * len(clients)
    if polling == "existing":
        for client, sequence in zip(clients, sequences, strict=True):
            snapshot = client.wait_for_ack(sequence)
            if snapshot.ack_seq != sequence:
                raise RuntimeError("command acknowledgement timed out")
    while pending:
        for index in tuple(pending):
            if polling == "ready":
                block = clients[index].block
                if not (block.ackSeq == sequences[index]
                        and block.currentFrame == frame
                        and block.runState == RUN_STATE_PAUSED):
                    continue
            snapshot = clients[index].snapshot()
            if snapshot.game_frame > frame:
                raise RuntimeError(f"advanced past expected frame {frame}")
            if (snapshot.ack_seq == sequences[index]
                    and snapshot.game_frame == frame
                    and snapshot.run_state_name == "PAUSED"):
                snapshots[index] = snapshot
                pending.remove(index)
        if pending:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"frame {frame} timed out: workers {pending}")
            if polling == "existing":
                time.sleep(0.001)
    return snapshots


def benchmark(config):
    workers = config["workers"]
    episodes = config["episodes"]
    frames = config["frames"]
    polling = config["polling"]
    seed = config["seed"]
    if workers < 1 or episodes < 1 or not 1 <= frames <= 4096:
        raise ValueError("workers/episodes must be positive; frames must be 1..4096")
    if polling not in {"existing", "busy", "ready"}:
        raise ValueError("polling must be existing, busy or ready")
    sokurl._validate_game()
    processes, clients, buffers = [], [], []
    launch_started = time.perf_counter()
    try:
        initial_hashes = []
        processes = sokurl._launch_vs_group_from_title(
            workers, 180.0, headless=True, unlimited=True, seed=seed, pause_at_start=True,
        )
        for index, process in enumerate(processes):
            client = BridgeClient(process.pid)
            clients.append(client)
            initial_hashes.append(wait_for_frame_zero(client, process.pid).stateHash)
            buffers.append((ctypes.c_ubyte * (FRAME_RING_CAPACITY * FRAME_SIZE))())
            print(f"ready {index + 1}/{workers} pid={process.pid}", flush=True)
        if len(set(initial_hashes)) != 1:
            raise RuntimeError("workers disagree at frame zero")
        launch_seconds = time.perf_counter() - launch_started
        trace = fixed_trace(frames)
        episode_reports = []
        reset_seconds = []
        batch_latencies = []
        expected_hashes = []
        sampled_started = time.perf_counter()
        for episode in range(episodes):
            if episode:
                reset_started = time.perf_counter()
                # The native bridge explicitly rejects GotoFrame. Restart the
                # original game instead of treating a partial state patch as reset.
                for client in clients:
                    client.close()
                clients.clear()
                for process in processes:
                    if process.is_running():
                        sokurl.shutdown(5.0, process.pid)
                processes.clear()
                processes = sokurl._launch_vs_group_from_title(
                    workers, 180.0, headless=True, unlimited=True,
                    seed=seed, pause_at_start=True,
                )
                for process, initial in zip(processes, initial_hashes, strict=True):
                    client = BridgeClient(process.pid)
                    clients.append(client)
                    if wait_for_frame_zero(client, process.pid).stateHash != initial:
                        raise RuntimeError("restart changed initial state hash")
                reset_seconds.append(time.perf_counter() - reset_started)
            started = time.perf_counter()
            for frame, (p1, p2) in enumerate(trace, 1):
                batch_started = time.perf_counter()
                sequences = [client.step_with_inputs(p1, p2) for client in clients]
                snapshots = wait_group(clients, sequences, frame, polling)
                hashes = [s.latest.stateHash for s in snapshots]
                if len(set(hashes)) != 1:
                    raise RuntimeError(f"worker state mismatch at frame {frame}")
                if episode == 0:
                    expected_hashes.append(hashes[0])
                elif hashes[0] != expected_hashes[frame - 1]:
                    raise RuntimeError(f"reset replay mismatch at frame {frame}")
                for client, buffer in zip(clients, buffers, strict=True):
                    _drain_fast(client, buffer)
                batch_latencies.append(time.perf_counter() - batch_started)
            elapsed = time.perf_counter() - started
            episode_reports.append({
                "seconds": elapsed, "aggregate_steps_per_second": workers * frames / elapsed,
                "final_hash": f"{snapshots[0].latest.stateHash:016X}",
                "p1_hp": snapshots[0].latest.p1.hp, "p2_hp": snapshots[0].latest.p2.hp,
            })
            print(f"episode {episode + 1}/{episodes}: {workers * frames / elapsed:.2f} steps/s", flush=True)
        sampled_seconds = time.perf_counter() - sampled_started
        dropped = [client.snapshot().dropped_frames for client in clients]
        if any(dropped):
            raise RuntimeError(f"dropped frames: {dropped}")
        batch_latencies.sort()
        return {
            "success": True, "config": config, "launch_seconds": launch_seconds,
            "sampled_seconds_including_resets": sampled_seconds,
            "aggregate_steps_per_second_including_resets": workers * frames * episodes / sampled_seconds,
            "simulation_steps": workers * frames * episodes,
            "batch_latency_median_ms": statistics.median(batch_latencies) * 1000,
            "batch_latency_p95_ms": batch_latencies[int(len(batch_latencies) * .95)] * 1000,
            "reset_batch_seconds": reset_seconds, "episodes": episode_reports,
            "dropped_frames": dropped, "all_frame_hashes_match": True,
            "policy_inference_included": False,
            "reset_method": "process_restart",
        }
    finally:
        for client in clients:
            client.close()
        for process in processes:
            if process.is_running():
                sokurl.shutdown(5.0, process.pid)


def main():
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    destination = Path(config["output"])
    try:
        report = benchmark(config)
    except Exception as error:
        report = {"success": False, "config": config, "error": repr(error)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

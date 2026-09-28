from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from bridge_shared import BridgeClient, BridgeUnavailable, FRAME_RING_CAPACITY, RawFrameState
from headless_validation import wait_for_frame_zero
import sokurl


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs" / "validation"


FRAME_SIZE = ctypes.sizeof(RawFrameState)


def _drain_fast(client: BridgeClient, buffer) -> int:
    block = client.block
    write = block.ringWriteSeq
    read = block.ringReadSeq
    available = (write - read) & 0xFFFFFFFF
    if available > FRAME_RING_CAPACITY:
        raise BridgeUnavailable("ring sequence accounting is invalid")
    if not available:
        return 0
    first_index = read % FRAME_RING_CAPACITY
    first_count = min(available, FRAME_RING_CAPACITY - first_index)
    ctypes.memmove(
        ctypes.addressof(buffer),
        ctypes.addressof(client.mapping.frames[first_index]),
        first_count * FRAME_SIZE,
    )
    remaining = available - first_count
    if remaining:
        ctypes.memmove(
            ctypes.addressof(buffer) + first_count * FRAME_SIZE,
            ctypes.addressof(client.mapping.frames[0]),
            remaining * FRAME_SIZE,
        )
    block.ringReadSeq = write
    return available


def _wait_ack(client: BridgeClient, sequence: int, buffer=None, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = client.snapshot()
        if buffer is not None:
            _drain_fast(client, buffer)
        if snapshot.ack_seq == sequence:
            return snapshot
    raise RuntimeError(f"PID {client.pid}: command {sequence} was not acknowledged")


def benchmark(worker_count: int, duration: float, seed: int, mode: str) -> dict[str, object]:
    processes: list[psutil.Process] = []
    clients: list[BridgeClient] = []
    buffers = []
    stop_consumers = threading.Event()
    consumer_threads: list[threading.Thread] = []
    try:
        for _ in range(worker_count):
            process = sokurl._launch_vs_from_title(
                35.0,
                headless=mode in {"headless", "unlimited"},
                unlimited=mode == "unlimited",
                seed=seed,
                pause_at_start=True,
            )
            processes.append(process)
            client = BridgeClient(process.pid)
            clients.append(client)
            buffers.append((ctypes.c_ubyte * (FRAME_RING_CAPACITY * FRAME_SIZE))())
            wait_for_frame_zero(client, process.pid)

        psutil.cpu_percent(interval=None)
        for process in processes:
            process.cpu_percent(interval=None)

        max_ring_backlog = [0] * worker_count
        consumer_errors: list[str] = []

        def consume(index: int) -> None:
            client = clients[index]
            buffer = buffers[index]
            try:
                while not stop_consumers.is_set():
                    backlog = (
                        client.block.ringWriteSeq - client.block.ringReadSeq
                    ) & 0xFFFFFFFF
                    max_ring_backlog[index] = max(max_ring_backlog[index], backlog)
                    if not _drain_fast(client, buffer):
                        time.sleep(0.0005)
            except Exception as error:
                consumer_errors.append(f"PID {client.pid}: {error}")

        consumer_threads = [
            threading.Thread(target=consume, args=(index,), daemon=True)
            for index in range(worker_count)
        ]
        for thread in consumer_threads:
            thread.start()

        sequences = [client.run() for client in clients]
        starts = []
        for client, sequence in zip(clients, sequences, strict=True):
            snapshot = _wait_ack(client, sequence)
            starts.append((snapshot.game_frame, time.perf_counter()))

        deadline = time.perf_counter() + duration
        system_cpu_samples: list[float] = []
        next_cpu_sample = time.perf_counter()
        while time.perf_counter() < deadline:
            if time.perf_counter() >= next_cpu_sample:
                system_cpu_samples.append(psutil.cpu_percent(interval=None))
                next_cpu_sample += 0.1
            if consumer_errors:
                raise RuntimeError(consumer_errors[0])
            time.sleep(0.001)

        pause_sequences = [client.pause() for client in clients]
        ends = []
        for client, sequence in zip(clients, pause_sequences, strict=True):
            snapshot = _wait_ack(client, sequence)
            ends.append((snapshot.game_frame, time.perf_counter(), snapshot.dropped_frames))
        stop_consumers.set()
        for thread in consumer_threads:
            thread.join(timeout=2.0)

        workers = []
        for index, (process, start, end) in enumerate(
            zip(processes, starts, ends, strict=True)
        ):
            elapsed = end[1] - start[1]
            frames = end[0] - start[0]
            workers.append({
                "pid": process.pid,
                "start_frame": start[0],
                "end_frame": end[0],
                "simulation_frames": frames,
                "elapsed_seconds": elapsed,
                "sim_fps": frames / elapsed,
                "cpu_percent_one_core_100": process.cpu_percent(interval=None),
                "dropped_frames": end[2],
                "max_ring_backlog": max_ring_backlog[index],
                "alive": process.is_running(),
            })

        per_instance = [float(worker["sim_fps"]) for worker in workers]
        recording_stable = all(worker["dropped_frames"] == 0 for worker in workers)
        return {
            "success": all(worker["alive"] for worker in workers) and recording_stable,
            "recording_stable": recording_stable,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "worker_count": worker_count,
            "mode": mode,
            "requested_duration_seconds": duration,
            "per_instance_sim_fps": per_instance,
            "mean_sim_fps": statistics.fmean(per_instance),
            "min_sim_fps": min(per_instance),
            "max_sim_fps": max(per_instance),
            "aggregate_sim_fps": sum(per_instance),
            "system_cpu_percent_mean": statistics.fmean(system_cpu_samples),
            "system_cpu_percent_max": max(system_cpu_samples),
            "gpu_usage": None,
            "gpu_note": "Per-process Direct3D utilization was not available from psutil.",
            "workers": workers,
        }
    finally:
        stop_consumers.set()
        for thread in consumer_threads:
            thread.join(timeout=2.0)
        for client in clients:
            client.close()
        for process in processes:
            if process.is_running():
                sokurl.shutdown(5.0, process.pid)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark headless unlimited VS workers")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--mode", choices=("normal", "headless", "unlimited"), default="unlimited")
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x534F4B55)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.duration <= 0:
        parser.error("--duration must be positive")

    try:
        report = benchmark(args.workers, args.duration, args.seed, args.mode)
    except Exception as error:
        report = {"success": False, "error": str(error), "worker_count": args.workers}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / (
        f"pacing-benchmark-{args.mode}-{args.workers}w-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="ascii")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"report={path}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

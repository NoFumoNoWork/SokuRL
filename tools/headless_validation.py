from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bridge_shared import ACTION_INPUTS, BridgeClient, calculate_state_hash
from frame_validation import copy_state, state_diff
import sokurl


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs" / "validation"
InputTuple = tuple[int, int, int, int, int, int, int, int]
NEUTRAL: InputTuple = ACTION_INPUTS["NEUTRAL"]


def _button(name: str, horizontal: int = 0, vertical: int = 0) -> InputTuple:
    values = list(NEUTRAL)
    values[0] = horizontal
    values[1] = vertical
    if name:
        values[{"A": 2, "B": 3, "C": 4, "D": 5}[name]] = 1
    return tuple(values)  # type: ignore[return-value]


def fixed_trace(frames: int) -> list[tuple[InputTuple, InputTuple]]:
    trace: list[tuple[InputTuple, InputTuple]] = []

    def add(count: int, p1: InputTuple = NEUTRAL, p2: InputTuple = NEUTRAL) -> None:
        trace.extend((p1, p2) for _ in range(count))

    add(90)
    add(180, ACTION_INPUTS["RIGHT"], ACTION_INPUTS["LEFT"])
    for cycle in range(12):
        button = ("A", "B", "C")[cycle % 3]
        add(3, _button(button), _button(button))
        add(47)
    add(300, ACTION_INPUTS["RIGHT"], ACTION_INPUTS["RIGHT"])
    for cycle in range(18):
        button = ("B", "C", "A")[cycle % 3]
        p1_direction = 1 if cycle % 2 == 0 else 0
        p2_direction = -1 if cycle % 2 == 1 else 0
        add(3, _button(button, p1_direction), _button(button, p2_direction))
        add(37)
    add(180, ACTION_INPUTS["UP_RIGHT"], ACTION_INPUTS["UP_LEFT"])
    for cycle in range(12):
        button = ("A", "B", "C")[cycle % 3]
        add(3, _button(button), _button(button))
        add(42)
    add(max(0, frames - len(trace)))
    return trace[:frames]


def wait_for_frame_zero(client: BridgeClient, pid: int, timeout: float = 35.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = client.snapshot()
        if (
            snapshot.in_gameplay
            and snapshot.checkpoint_valid
            and snapshot.game_frame == 0
            and snapshot.run_state_name == "PAUSED"
        ):
            state = copy_state(snapshot.latest)
            if state.stateHash != calculate_state_hash(state):
                raise RuntimeError(f"PID {pid}: native/Python frame-zero hash mismatch")
            client.drain_frames()
            return state
        time.sleep(0.005)
    raise RuntimeError(f"PID {pid}: timed out waiting for paused VS frame zero; "
                       f"gameplay={snapshot.in_gameplay} checkpoint={snapshot.checkpoint_valid} "
                       f"frame={snapshot.game_frame} run_state={snapshot.run_state_name} "
                       f"scene={snapshot.latest.sceneId} result={snapshot.result_code}")


def step_group(
    clients: list[BridgeClient],
    p1: InputTuple,
    p2: InputTuple,
    expected_frame: int,
    timeout: float = 3.0,
):
    sequences = [client.step_with_inputs(p1, p2) for client in clients]
    deadline = time.monotonic() + timeout
    snapshots = [None] * len(clients)
    while time.monotonic() < deadline:
        for index, client in enumerate(clients):
            snapshot = client.snapshot()
            if snapshot.game_frame > expected_frame:
                raise RuntimeError(
                    f"PID {client.pid}: advanced past frame {expected_frame} "
                    f"to {snapshot.game_frame}"
                )
            if (
                snapshot.ack_seq == sequences[index]
                and snapshot.game_frame == expected_frame
                and snapshot.run_state_name == "PAUSED"
            ):
                snapshots[index] = snapshot
        if all(snapshot is not None for snapshot in snapshots):
            return tuple(copy_state(snapshot.latest) for snapshot in snapshots)
        time.sleep(0.001)
    raise RuntimeError(f"timed out stepping paired frame {expected_frame}")


def _coverage_update(coverage: dict[str, object], state) -> None:
    coverage["max_p1_objects"] = max(int(coverage["max_p1_objects"]), state.p1ObjectCount)
    coverage["max_p2_objects"] = max(int(coverage["max_p2_objects"]), state.p2ObjectCount)
    coverage["min_p1_hp"] = min(int(coverage["min_p1_hp"]), state.p1.hp)
    coverage["min_p2_hp"] = min(int(coverage["min_p2_hp"]), state.p2.hp)
    coverage["max_hitstop"] = max(
        int(coverage["max_hitstop"]), state.p1.hitstop, state.p2.hitstop
    )
    coverage["max_untech"] = max(int(coverage["max_untech"]), state.p1.untech, state.p2.untech)
    coverage["airborne_seen"] = bool(coverage["airborne_seen"] or state.p1.airborne or state.p2.airborne)
    coverage["min_x"] = min(float(coverage["min_x"]), state.p1.x, state.p2.x)
    coverage["max_x"] = max(float(coverage["max_x"]), state.p1.x, state.p2.x)
    actions = coverage["actions"]
    assert isinstance(actions, set)
    actions.add((state.p1.actionId, state.p2.actionId))


def run(frames: int, seed: int, include_unlimited: bool = False) -> dict[str, object]:
    sokurl._validate_game()
    processes = []
    clients: list[BridgeClient] = []
    started = time.perf_counter()
    try:
        normal_process = sokurl._launch_vs_from_title(
            35.0, seed=seed, pause_at_start=True
        )
        processes.append(normal_process)
        headless_process = sokurl._launch_vs_from_title(
            35.0, headless=True, seed=seed, pause_at_start=True
        )
        processes.append(headless_process)
        if include_unlimited:
            unlimited_process = sokurl._launch_vs_from_title(
                35.0, headless=True, unlimited=True, seed=seed, pause_at_start=True
            )
            processes.append(unlimited_process)
        clients = [BridgeClient(process.pid) for process in processes]

        initial = [
            wait_for_frame_zero(client, process.pid)
            for client, process in zip(clients, processes, strict=True)
        ]
        if any(state.stateHash != initial[0].stateHash for state in initial[1:]):
            return {
                "success": False,
                "first_divergent_frame": 0,
                "hashes": [f"{state.stateHash:016X}" for state in initial],
                "diffs": [state_diff(initial[0], state) for state in initial[1:]],
            }

        coverage: dict[str, object] = {
            "max_p1_objects": 0,
            "max_p2_objects": 0,
            "min_p1_hp": initial[0].p1.hp,
            "min_p2_hp": initial[0].p2.hp,
            "max_hitstop": 0,
            "max_untech": 0,
            "airborne_seen": False,
            "min_x": min(initial[0].p1.x, initial[0].p2.x),
            "max_x": max(initial[0].p1.x, initial[0].p2.x),
            "actions": set(),
        }
        trace = fixed_trace(frames)
        for frame, (p1, p2) in enumerate(trace, 1):
            states = step_group(clients, p1, p2, frame)
            for index, state in enumerate(states):
                if state.stateHash != calculate_state_hash(state):
                    raise RuntimeError(
                        f"mode {index} native/Python hash mismatch at frame {frame}"
                    )
            normal_state = states[0]
            divergent = next(
                (index for index, state in enumerate(states[1:], 1)
                 if state.stateHash != normal_state.stateHash),
                None,
            )
            if divergent is not None:
                return {
                    "success": False,
                    "seed": seed,
                    "frames_requested": frames,
                    "first_divergent_frame": frame,
                    "normal_hash": f"{normal_state.stateHash:016X}",
                    "divergent_mode": ("headless", "unlimited")[divergent - 1],
                    "divergent_hash": f"{states[divergent].stateHash:016X}",
                    "p1_input": p1,
                    "p2_input": p2,
                    "scene": normal_state.sceneId,
                    "mode": normal_state.battleMode,
                    "battle_manager_update_count": frame,
                    "diff": state_diff(normal_state, states[divergent]),
                }
            _coverage_update(coverage, normal_state)
            if frame % 128 == 0:
                for client in clients:
                    client.drain_frames()

        actions = coverage.pop("actions")
        assert isinstance(actions, set)
        coverage["distinct_action_pairs"] = len(actions)
        elapsed = time.perf_counter() - started
        return {
            "success": True,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "frames_compared": frames,
            "modes": ["normal", "headless"] + (["unlimited"] if include_unlimited else []),
            "pids": [process.pid for process in processes],
            "initial_hash": f"{initial[0].stateHash:016X}",
            "final_hash": f"{normal_state.stateHash:016X}",
            "elapsed_seconds": elapsed,
            "paired_sim_frames_per_second": frames / elapsed,
            "coverage": coverage,
            "dropped_frames": {
                mode: client.snapshot().dropped_frames
                for mode, client in zip(
                    ["normal", "headless"] + (["unlimited"] if include_unlimited else []),
                    clients,
                    strict=True,
                )
            },
        }
    finally:
        for client in clients:
            client.close()
        for process in processes:
            if process.is_running():
                sokurl.shutdown(5.0, process.pid)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare rendered and M2-A headless VS state")
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x534F4B55)
    parser.add_argument("--unlimited", action="store_true", help="add headless unlimited mode")
    args = parser.parse_args()
    if not 1 <= args.frames <= 12_000:
        parser.error("--frames must be between 1 and 12000")

    try:
        report = run(args.frames, args.seed, args.unlimited)
    except Exception as error:
        report = {"success": False, "error": str(error)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = "unlimited-equivalence" if args.unlimited else "headless-equivalence"
    path = REPORT_DIR / f"{prefix}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="ascii")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"report={path}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

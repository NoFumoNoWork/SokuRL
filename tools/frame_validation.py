from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import psutil

import sokurl
from bridge_shared import (
    ACTION_INPUTS,
    BridgeClient,
    BridgeUnavailable,
    LogicalInput,
    RawFrameState,
    calculate_state_hash,
)


InputTuple = tuple[int, int, int, int, int, int, int, int]
NEUTRAL: InputTuple = ACTION_INPUTS["NEUTRAL"]
ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs" / "validation"


def copy_state(state: RawFrameState) -> RawFrameState:
    result = RawFrameState()
    ctypes.memmove(ctypes.addressof(result), ctypes.addressof(state), ctypes.sizeof(result))
    return result


def input_tuple(value: LogicalInput) -> InputTuple:
    return tuple(
        getattr(value, name)
        for name in ("horizontalAxis", "verticalAxis", "a", "b", "c", "d", "changeCard", "spellcard")
    )


def flatten_ctypes(value: object, prefix: str = "") -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    if isinstance(value, ctypes.Array):
        for index, item in enumerate(value):
            result.update(flatten_ctypes(item, f"{prefix}[{index}]"))
        return result
    if isinstance(value, ctypes.Structure):
        for name, _ in value._fields_:
            child = getattr(value, name)
            child_prefix = f"{prefix}.{name}" if prefix else name
            result.update(flatten_ctypes(child, child_prefix))
        return result
    result[prefix] = value  # type: ignore[assignment]
    return result


def state_diff(expected: RawFrameState, actual: RawFrameState) -> list[dict[str, object]]:
    left = flatten_ctypes(expected)
    right = flatten_ctypes(actual)
    return [
        {"field": name, "expected": left[name], "actual": right[name]}
        for name in left
        if left[name] != right[name]
    ]


SIMPLE_FIELDS = {
    "timeElapsedRaw", "activeWeather", "displayedWeather", "weatherCounter",
    *(f"{player}.{name}" for player in ("p1", "p2") for name in (
        "x", "y", "speedX", "speedY", "facing", "hp", "spirit", "maxSpirit",
        "cardGauge", "cardCount",
    )),
}


def complex_state_diff(expected: RawFrameState, actual: RawFrameState) -> list[dict[str, object]]:
    return [
        difference for difference in state_diff(expected, actual)
        if difference["field"] not in SIMPLE_FIELDS and difference["field"] != "stateHash"
    ]


@dataclass(frozen=True)
class InputPair:
    p1: InputTuple
    p2: InputTuple = NEUTRAL


class PracticeInstance:
    def __init__(self, process: psutil.Process, client: BridgeClient, confirmations: int) -> None:
        self.process = process
        self.client = client
        self.confirmations = confirmations

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def state(self) -> RawFrameState:
        return copy_state(self.client.snapshot().latest)

    def step(self, inputs: InputPair, timeout: float = 3.0) -> RawFrameState:
        before = self.client.snapshot().game_frame
        sequence = self.client.step_with_inputs(inputs.p1, inputs.p2)
        acknowledged = self.client.wait_for_ack(sequence, timeout=timeout)
        if acknowledged.ack_seq != sequence:
            raise RuntimeError(f"PID {self.pid}: step command was not acknowledged")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.client.snapshot()
            if snapshot.game_frame == before + 1 and snapshot.run_state_name == "PAUSED":
                state = copy_state(snapshot.latest)
                if state.stateHash != calculate_state_hash(state):
                    raise RuntimeError(f"PID {self.pid}: native/Python hash mismatch at frame {state.frameId}")
                self.client.drain_frames()
                return state
            if snapshot.game_frame > before + 1:
                raise RuntimeError(
                    f"PID {self.pid}: one-frame command advanced {snapshot.game_frame - before} frames"
                )
            time.sleep(0.005)
        raise RuntimeError(f"PID {self.pid}: timed out stepping frame {before + 1}")

    def step_native(self, timeout: float = 3.0) -> RawFrameState:
        before = self.client.snapshot().game_frame
        sequence = self.client.step(1)
        acknowledged = self.client.wait_for_ack(sequence, timeout=timeout)
        if acknowledged.ack_seq != sequence:
            raise RuntimeError(f"PID {self.pid}: native step command was not acknowledged")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.client.snapshot()
            if snapshot.game_frame == before + 1 and snapshot.run_state_name == "PAUSED":
                state = copy_state(snapshot.latest)
                if state.stateHash != calculate_state_hash(state):
                    raise RuntimeError(
                        f"PID {self.pid}: native/Python hash mismatch at frame {state.frameId}"
                    )
                self.client.drain_frames()
                return state
            if snapshot.game_frame > before + 1:
                raise RuntimeError(
                    f"PID {self.pid}: native one-frame command advanced "
                    f"{snapshot.game_frame - before} frames"
                )
            time.sleep(0.005)
        raise RuntimeError(f"PID {self.pid}: timed out native-stepping frame {before + 1}")

    def apply_simple(self, expected: RawFrameState, timeout: float = 3.0) -> RawFrameState:
        sequence = self.client.apply_simple_state(expected)
        acknowledged = self.client.wait_for_ack(sequence, timeout=timeout)
        if acknowledged.ack_seq != sequence or acknowledged.result_name != "COMPLETE":
            raise RuntimeError(
                f"PID {self.pid}: simple-state patch failed: {acknowledged.result_name}"
            )
        state = copy_state(acknowledged.latest)
        if state.stateHash != calculate_state_hash(state):
            raise RuntimeError(
                f"PID {self.pid}: hash mismatch after simple patch at frame {state.frameId}"
            )
        self.client.drain_frames()
        return state

    def close(self) -> None:
        self.client.close()
        if self.process.is_running():
            sokurl.shutdown(5.0, self.pid)


def launch_checkpoint(seed: int | None = None, timeout: float = 35.0) -> PracticeInstance:
    sokurl._validate_game()
    process = psutil.Process(subprocess.Popen([str(sokurl.GAME_EXE)], cwd=sokurl.GAME_DIR).pid)
    client: BridgeClient | None = None
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if not process.is_running():
                raise RuntimeError(f"th123 exited before preset readiness (PID {process.pid})")
            state = sokurl._runtime_state(process)
            if state.health == "PRACTICE_PRESET_READY":
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f"timed out waiting for Practice preset (PID {process.pid})")

        client = BridgeClient(process.pid)
        sequence = client.establish_checkpoint(seed)
        acknowledged = client.wait_for_ack(sequence)
        if acknowledged.ack_seq != sequence or acknowledged.result_name != "ACCEPTED":
            raise RuntimeError(
                f"PID {process.pid}: checkpoint arm failed: {acknowledged.result_name}"
            )

        confirmations = 0
        next_confirmation = 0.0
        while time.monotonic() < deadline:
            runtime = sokurl._runtime_state(process)
            snapshot = client.snapshot()
            if runtime.health == "PRACTICE_READY" and snapshot.checkpoint_valid:
                if snapshot.game_frame != 0 or snapshot.run_state_name != "PAUSED":
                    raise RuntimeError(
                        f"PID {process.pid}: checkpoint was not frozen at frame 0: "
                        f"frame={snapshot.game_frame} state={snapshot.run_state_name}"
                    )
                client.drain_frames()
                return PracticeInstance(process, client, confirmations)
            if runtime.health == "PRACTICE_PRESET_READY" and time.monotonic() >= next_confirmation:
                if confirmations >= 12:
                    raise RuntimeError("menu confirmation limit reached")
                sequence = client.menu_confirm()
                acknowledged = client.wait_for_ack(sequence)
                if acknowledged.ack_seq != sequence:
                    raise RuntimeError("menu confirmation was not acknowledged")
                confirmations += 1
                next_confirmation = time.monotonic() + 0.6
            time.sleep(0.05)
        raise RuntimeError(f"timed out waiting for paused frame-zero checkpoint (PID {process.pid})")
    except Exception:
        if client is not None:
            client.close()
        if process.is_running():
            sokurl.shutdown(3.0, process.pid)
        raise


def append_actions(target: list[InputPair], action: str, count: int = 1) -> None:
    target.extend(InputPair(ACTION_INPUTS[action]) for _ in range(count))


def scripted_inputs() -> list[InputPair]:
    result: list[InputPair] = []
    append_actions(result, "NEUTRAL", 300)
    append_actions(result, "RIGHT", 30)
    append_actions(result, "NEUTRAL", 20)
    for _ in range(3):
        append_actions(result, "A")
        append_actions(result, "NEUTRAL", 35)
    append_actions(result, "B")
    append_actions(result, "NEUTRAL", 100)
    append_actions(result, "B")
    append_actions(result, "NEUTRAL", 130)
    append_actions(result, "C")
    append_actions(result, "NEUTRAL", 130)
    append_actions(result, "C")
    append_actions(result, "NEUTRAL", 220)
    return result


def record_trace(inputs: Iterable[InputPair]) -> tuple[int, list[RawFrameState], list[InputPair]]:
    instance = launch_checkpoint()
    try:
        states = [instance.state]
        applied: list[InputPair] = []
        for pair in inputs:
            state = instance.step(pair)
            states.append(state)
            applied.append(InputPair(input_tuple(state.p1.input), input_tuple(state.p2.input)))
        return instance.confirmations, states, applied
    finally:
        instance.close()


def replay_to_target(
    seed: int,
    recorded: list[RawFrameState],
    inputs: list[InputPair],
    target: int,
) -> tuple[RawFrameState, int | None, list[dict[str, object]]]:
    if not 0 <= target < len(recorded):
        raise ValueError(f"target {target} is outside recorded trace")
    instance = launch_checkpoint(seed)
    try:
        initial_raw = instance.state
        differences = complex_state_diff(recorded[0], initial_raw)
        if differences:
            return initial_raw, 0, differences
        initial = instance.apply_simple(recorded[0])
        differences = state_diff(recorded[0], initial)
        if differences:
            return initial, 0, differences
        current = initial
        for frame in range(1, target + 1):
            simulated = instance.step(inputs[frame - 1])
            differences = complex_state_diff(recorded[frame], simulated)
            if differences:
                return simulated, frame, differences
            current = instance.apply_simple(recorded[frame])
            differences = state_diff(recorded[frame], current)
            if differences:
                return current, frame, differences
        return current, None, []
    finally:
        instance.close()


def projectile_targets(states: list[RawFrameState]) -> list[int]:
    active = [index for index, state in enumerate(states) if state.p1ObjectCount]
    if not active:
        raise RuntimeError("script produced no P1 projectile/object state")
    peak = max(active, key=lambda index: states[index].p1ObjectCount)
    first = active[0]
    last = active[-1]
    after_despawn = min(last + 1, len(states) - 1)
    candidates = [first, peak, last, after_despawn]
    return list(dict.fromkeys(candidates))


def verify_two_instance_isolation(seed: int) -> dict[str, object]:
    first = launch_checkpoint(seed)
    second: PracticeInstance | None = None
    try:
        second = launch_checkpoint(seed)
        before = second.client.snapshot().game_frame
        first.step(InputPair(ACTION_INPUTS["RIGHT"]))
        after = second.client.snapshot().game_frame
        return {
            "pids": [first.pid, second.pid],
            "mapping_names_distinct": first.pid != second.pid,
            "second_frame_before": before,
            "second_frame_after": after,
            "isolated": before == after == 0,
        }
    finally:
        if second is not None:
            second.close()
        first.close()


def run_validation() -> dict[str, object]:
    started = datetime.now(timezone.utc).isoformat()
    confirmations, recorded, inputs = record_trace(scripted_inputs())
    seed = recorded[0].randomSeed
    targets = projectile_targets(recorded)
    report: dict[str, object] = {
        "started_utc": started,
        "seed": seed,
        "recorded_frames": len(recorded),
        "confirmations": confirmations,
        "max_p1_objects": max(state.p1ObjectCount for state in recorded),
        "max_p2_objects": max(state.p2ObjectCount for state in recorded),
        "object_overflow": any(
            state.p1ObjectOverflow or state.p2ObjectOverflow for state in recorded
        ),
        "targets": targets,
        "target_results": [],
    }

    for target in targets:
        reconstructed, divergence, differences = replay_to_target(seed, recorded, inputs, target)
        result = {
            "target": target,
            "expected_hash": f"{recorded[target].stateHash:016X}",
            "actual_hash": f"{reconstructed.stateHash:016X}",
            "first_divergent_frame": divergence,
            "differences": differences[:100],
        }
        report["target_results"].append(result)  # type: ignore[union-attr]
        if divergence is not None:
            report["success"] = False
            return report

    middle = targets[len(targets) // 2]
    navigation = [middle, middle - 1, middle, targets[0], targets[-1], targets[0]]
    navigation_results = []
    for target in navigation:
        reconstructed, divergence, differences = replay_to_target(seed, recorded, inputs, target)
        navigation_results.append({
            "target": target,
            "hash": f"{reconstructed.stateHash:016X}",
            "expected_hash": f"{recorded[target].stateHash:016X}",
            "first_divergent_frame": divergence,
            "differences": differences[:100],
        })
        if divergence is not None:
            report["navigation_results"] = navigation_results
            report["success"] = False
            return report
    report["navigation_results"] = navigation_results
    report["minus_one_plus_one_identity"] = (
        navigation_results[0]["hash"] == navigation_results[2]["hash"]
    )

    isolation = verify_two_instance_isolation(seed)
    report["two_instance_isolation"] = isolation
    report["success"] = bool(
        report["minus_one_plus_one_identity"] and isolation["isolated"]
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runtime deterministic frame reconstruction validator"
    )
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_validation()
    except (BridgeUnavailable, OSError, psutil.Error, RuntimeError, ValueError) as error:
        report = {"success": False, "error": str(error)}
    path = args.report
    if path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = REPORT_DIR / f"frame-validation-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="ascii")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"report={path}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

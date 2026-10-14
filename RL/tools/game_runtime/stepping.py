"""Step an owned paused game and verify the native frame and state hash."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import time

import psutil
import sokurl
from bridge_shared import ACTION_INPUTS, BridgeClient, LogicalInput, RawFrameState, calculate_state_hash

InputTuple = tuple[int, int, int, int, int, int, int, int]
NEUTRAL: InputTuple = ACTION_INPUTS["NEUTRAL"]


def copy_state(state: RawFrameState) -> RawFrameState:
    result = RawFrameState()
    ctypes.memmove(ctypes.addressof(result), ctypes.addressof(state), ctypes.sizeof(result))
    return result


def input_tuple(value: LogicalInput) -> InputTuple:
    return tuple(
        getattr(value, name)
        for name in ("horizontalAxis", "verticalAxis", "a", "b", "c", "d", "changeCard", "spellcard")
    )


@dataclass(frozen=True)
class InputPair:
    p1: InputTuple
    p2: InputTuple = NEUTRAL


class PausedGame:
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



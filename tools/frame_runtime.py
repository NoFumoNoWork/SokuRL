from __future__ import annotations

from dataclasses import dataclass


def goto_plan(current: int, target: int, recorded_frames: int, checkpoint_valid: bool) -> str:
    if target < 0:
        raise ValueError("target must be non-negative")
    if not checkpoint_valid:
        raise ValueError("checkpoint is not valid")
    if target >= recorded_frames:
        raise ValueError("target frame has not been recorded")
    return "already-there" if target == current else "restart-and-resimulate"


def step_back_target(current: int) -> int:
    if current <= 0:
        raise ValueError("already at frame 0")
    return current - 1


def first_hash_divergence(recorded: list[int], reconstructed: list[int]) -> int | None:
    for frame, (expected, actual) in enumerate(zip(recorded, reconstructed, strict=False)):
        if expected != actual:
            return frame
    if len(recorded) != len(reconstructed):
        return min(len(recorded), len(reconstructed))
    return None


@dataclass
class SimulationAccounting:
    frame: int = 0
    paused: bool = True

    def run_update(self) -> int:
        if not self.paused:
            self.frame += 1
        return self.frame

    def step(self, count: int) -> int:
        if count <= 0:
            raise ValueError("step count must be positive")
        self.paused = True
        self.frame += count
        return self.frame


@dataclass(frozen=True)
class CheckpointFingerprint:
    left_character: int
    right_character: int
    stage: int
    seed: int
    practice_settings: tuple[int, ...]

    def remains_valid(self, current: "CheckpointFingerprint") -> bool:
        return self == current


@dataclass
class RingAccounting:
    capacity: int
    write_sequence: int = 0
    read_sequence: int = 0
    dropped: int = 0

    def push(self) -> bool:
        if self.write_sequence - self.read_sequence >= self.capacity:
            self.dropped += 1
            return False
        self.write_sequence += 1
        return True

    def drain(self, count: int | None = None) -> int:
        available = self.write_sequence - self.read_sequence
        consumed = available if count is None else min(available, max(count, 0))
        self.read_sequence += consumed
        return consumed

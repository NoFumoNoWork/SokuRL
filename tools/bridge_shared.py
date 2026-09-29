from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from dataclasses import dataclass

MAPPING_NAME_FORMAT = r"Local\SokuRLBridge_{}"
CONTROL_MAGIC = 0x554B4F53
CONTROL_VERSION = 7
MAX_DURATION_FRAMES = 10_000
FRAME_RING_CAPACITY = 512
INPUT_HISTORY_CAPACITY = 4096
MAX_OBJECTS_PER_PLAYER = 64
NO_FRAME = (1 << 64) - 1


def wait_for_steps(clients, sequences, frames, timeout):
    """Wait for exact completed frames, including independently reset slots."""
    if not (len(clients) == len(sequences) == len(frames)) or timeout <= 0:
        raise ValueError("invalid step wait arguments")
    pending = set(range(len(clients)))
    snapshots = [None] * len(clients)
    deadline = time.monotonic() + timeout
    while pending:
        for index in tuple(pending):
            block = clients[index].block
            if block.currentFrame > frames[index]:
                raise RuntimeError("game advanced past requested frame")
            if (block.ackSeq == sequences[index] and block.currentFrame == frames[index]
                    and block.runState == 1):
                snapshot = clients[index].snapshot()
                if (snapshot.ack_seq == sequences[index] and snapshot.game_frame == frames[index]
                        and snapshot.run_state_name == "PAUSED"):
                    snapshots[index] = snapshot
                    pending.remove(index)
        if pending and time.monotonic() >= deadline:
            raise TimeoutError(f"simulation step timed out for workers {sorted(pending)}")
    return snapshots

COMMAND_INPUT = 1
COMMAND_RELEASE = 2
COMMAND_RUN = 3
COMMAND_PAUSE = 4
COMMAND_STEP_FRAMES = 5
COMMAND_ESTABLISH_CHECKPOINT = 6
COMMAND_GOTO_FRAME = 7
COMMAND_MENU_CONFIRM = 8
COMMAND_STEP_WITH_INPUTS = 9
COMMAND_APPLY_SIMPLE_STATE = 10
COMMAND_RESET_EPISODE = 11

RESULT_NAMES = {
    0: "IDLE", 1: "ACCEPTED", 2: "COMPLETE", 3: "RELEASED",
    4: "NOT_IN_GAMEPLAY", 5: "INVALID_COMMAND", 6: "NO_CHECKPOINT",
    7: "TARGET_UNAVAILABLE", 8: "RESTARTING", 9: "DIVERGED",
    10: "CHECKPOINT_INVALIDATED", 11: "HISTORY_FULL",
    12: "CHECKPOINT_RESTORE_UNSUPPORTED",
}
RUN_STATE_NAMES = {0: "RUNNING", 1: "PAUSED", 2: "STEPPING", 3: "RECONSTRUCTING"}
VALIDATION_NAMES = {0: "UNKNOWN", 1: "YES", 2: "NO"}

ACTION_INPUTS = {
    "NEUTRAL": (0, 0, 0, 0, 0, 0, 0, 0),
    "LEFT": (-1, 0, 0, 0, 0, 0, 0, 0),
    "RIGHT": (1, 0, 0, 0, 0, 0, 0, 0),
    "UP": (0, -1, 0, 0, 0, 0, 0, 0),
    "DOWN": (0, 1, 0, 0, 0, 0, 0, 0),
    "UP_LEFT": (-1, -1, 0, 0, 0, 0, 0, 0),
    "UP_RIGHT": (1, -1, 0, 0, 0, 0, 0, 0),
    "DOWN_LEFT": (-1, 1, 0, 0, 0, 0, 0, 0),
    "DOWN_RIGHT": (1, 1, 0, 0, 0, 0, 0, 0),
    "A": (0, 0, 1, 0, 0, 0, 0, 0),
    "B": (0, 0, 0, 1, 0, 0, 0, 0),
    "C": (0, 0, 0, 0, 1, 0, 0, 0),
    "D": (0, 0, 0, 0, 0, 1, 0, 0),
    "LEFT_A": (-1, 0, 1, 0, 0, 0, 0, 0),
    "RIGHT_A": (1, 0, 1, 0, 0, 0, 0, 0),
    "DOWN_A": (0, 1, 1, 0, 0, 0, 0, 0),
    "UP_A": (0, -1, 1, 0, 0, 0, 0, 0),
    "LEFT_B": (-1, 0, 0, 1, 0, 0, 0, 0),
    "RIGHT_B": (1, 0, 0, 1, 0, 0, 0, 0),
    "DOWN_B": (0, 1, 0, 1, 0, 0, 0, 0),
    "UP_B": (0, -1, 0, 1, 0, 0, 0, 0),
    "LEFT_C": (-1, 0, 0, 0, 1, 0, 0, 0),
    "RIGHT_C": (1, 0, 0, 0, 1, 0, 0, 0),
    "DOWN_C": (0, 1, 0, 0, 1, 0, 0, 0),
    "UP_C": (0, -1, 0, 0, 1, 0, 0, 0),
}


class LogicalInput(ctypes.Structure):
    _pack_ = 4
    _fields_ = [(name, ctypes.c_int32) for name in (
        "horizontalAxis", "verticalAxis", "a", "b", "c", "d", "changeCard", "spellcard")]


class PlayerState(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("characterId", ctypes.c_uint32),
        ("x", ctypes.c_float), ("y", ctypes.c_float),
        ("speedX", ctypes.c_float), ("speedY", ctypes.c_float),
        ("facing", ctypes.c_int32), ("hp", ctypes.c_int32),
        ("spirit", ctypes.c_int32), ("maxSpirit", ctypes.c_int32),
        ("cardGauge", ctypes.c_uint32), ("cardCount", ctypes.c_uint32),
        ("handIds", ctypes.c_int32 * 5),
        ("actionId", ctypes.c_uint32), ("sequenceId", ctypes.c_uint32),
        ("subsequenceId", ctypes.c_uint32), ("animationFrame", ctypes.c_uint32),
        ("elapsedInSubsequence", ctypes.c_uint32), ("hitstop", ctypes.c_uint32),
        ("untech", ctypes.c_uint32), ("airborne", ctypes.c_uint32),
        ("frameFlags", ctypes.c_uint32), ("attackFlags", ctypes.c_uint32),
        ("objectCount", ctypes.c_uint32), ("input", LogicalInput),
    ]


class ObjectState(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("ownerIndex", ctypes.c_uint32), ("listIndex", ctypes.c_uint32),
        ("typeId", ctypes.c_uint32), ("actionId", ctypes.c_uint32),
        ("actionBlockId", ctypes.c_uint32), ("animationCounter", ctypes.c_uint32),
        ("animationSubFrame", ctypes.c_uint32), ("frameCount", ctypes.c_uint32),
        ("x", ctypes.c_float), ("y", ctypes.c_float),
        ("speedX", ctypes.c_float), ("speedY", ctypes.c_float),
        ("gravity", ctypes.c_float), ("direction", ctypes.c_int32),
        ("hp", ctypes.c_int32), ("hitstop", ctypes.c_uint32),
        ("hitBoxCount", ctypes.c_uint32), ("hurtBoxCount", ctypes.c_uint32),
        ("characterIndex", ctypes.c_uint32), ("isActive", ctypes.c_uint32),
    ]


class SimplePlayerState(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("x", ctypes.c_float), ("y", ctypes.c_float),
        ("speedX", ctypes.c_float), ("speedY", ctypes.c_float),
        ("facing", ctypes.c_int32), ("hp", ctypes.c_int32),
        ("spirit", ctypes.c_int32), ("maxSpirit", ctypes.c_int32),
        ("cardGauge", ctypes.c_uint32), ("cardCount", ctypes.c_uint32),
    ]


class SimpleStatePatch(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("timeElapsedRaw", ctypes.c_uint32),
        ("activeWeather", ctypes.c_uint32),
        ("displayedWeather", ctypes.c_uint32),
        ("weatherCounter", ctypes.c_uint32),
        ("p1", SimplePlayerState), ("p2", SimplePlayerState),
    ]


class ReconstructionFrame(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("p1Input", LogicalInput), ("p2Input", LogicalInput),
        ("simple", SimpleStatePatch), ("stateHash", ctypes.c_uint64),
    ]


class RawFrameState(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("frameId", ctypes.c_uint64), ("segmentId", ctypes.c_uint32),
        ("sceneId", ctypes.c_uint32), ("battleMode", ctypes.c_uint32),
        ("battleSubMode", ctypes.c_uint32), ("stageId", ctypes.c_uint32),
        ("roundId", ctypes.c_uint32), ("timeElapsedRaw", ctypes.c_uint32),
        ("activeWeather", ctypes.c_uint32), ("displayedWeather", ctypes.c_uint32),
        ("weatherCounter", ctypes.c_uint32), ("randomSeed", ctypes.c_uint32),
        ("p1", PlayerState), ("p2", PlayerState),
        ("p1ObjectCount", ctypes.c_uint32), ("p2ObjectCount", ctypes.c_uint32),
        ("p1ObjectOverflow", ctypes.c_uint32), ("p2ObjectOverflow", ctypes.c_uint32),
        ("p1Objects", ObjectState * MAX_OBJECTS_PER_PLAYER),
        ("p2Objects", ObjectState * MAX_OBJECTS_PER_PLAYER),
        ("stateHash", ctypes.c_uint64),
    ]


class ControlBlock(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("magic", ctypes.c_uint32), ("version", ctypes.c_uint32),
        ("structSize", ctypes.c_uint32), ("mappingSize", ctypes.c_uint32),
        ("commandSeq", ctypes.c_uint32), ("ackSeq", ctypes.c_uint32),
        ("commandType", ctypes.c_uint32), ("resultCode", ctypes.c_uint32),
        ("commandInput", LogicalInput), ("commandInputP2", LogicalInput),
        ("durationFrames", ctypes.c_uint32),
        ("inputFramesRemaining", ctypes.c_uint32), ("commandArgument", ctypes.c_uint64),
        ("statusSeq", ctypes.c_uint32), ("connected", ctypes.c_uint32),
        ("inGameplay", ctypes.c_uint32), ("runState", ctypes.c_uint32),
        ("checkpointValid", ctypes.c_uint32), ("validationState", ctypes.c_uint32),
        ("reconstructing", ctypes.c_uint32), ("stepsRemaining", ctypes.c_uint32),
        ("currentFrame", ctypes.c_uint64), ("recordedFrames", ctypes.c_uint64),
        ("lastVerifiedFrame", ctypes.c_uint64), ("firstDivergentFrame", ctypes.c_uint64),
        ("droppedFrames", ctypes.c_uint32), ("ringWriteSeq", ctypes.c_uint32),
        ("ringReadSeq", ctypes.c_uint32), ("ringCapacity", ctypes.c_uint32),
        ("commandPatch", SimpleStatePatch),
        ("latest", RawFrameState),
    ]


class BridgeMapping(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("control", ControlBlock),
        ("frames", RawFrameState * FRAME_RING_CAPACITY),
        ("history", ReconstructionFrame * INPUT_HISTORY_CAPACITY),
    ]


CONTROL_BLOCK_SIZE = ctypes.sizeof(ControlBlock)
MAPPING_SIZE = ctypes.sizeof(BridgeMapping)
assert ctypes.sizeof(LogicalInput) == 32
assert ctypes.sizeof(PlayerState) == 140
assert ctypes.sizeof(ObjectState) == 80
assert ctypes.sizeof(SimplePlayerState) == 40
assert ctypes.sizeof(SimpleStatePatch) == 96
assert ctypes.sizeof(ReconstructionFrame) == 168
assert ctypes.sizeof(RawFrameState) == 10596
assert CONTROL_BLOCK_SIZE == 10884
assert ControlBlock.currentFrame.offset == 144
assert ControlBlock.latest.offset == 288


class BridgeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BridgeSnapshot:
    command_seq: int
    ack_seq: int
    frames_remaining: int
    game_frame: int
    connected: bool
    in_gameplay: bool
    result_code: int
    run_state: int
    checkpoint_valid: bool
    validation_state: int
    reconstructing: bool
    steps_remaining: int
    recorded_frames: int
    dropped_frames: int
    last_verified_frame: int | None
    first_divergent_frame: int | None
    latest: RawFrameState

    @property
    def result_name(self) -> str:
        return RESULT_NAMES.get(self.result_code, f"UNKNOWN_{self.result_code}")

    @property
    def run_state_name(self) -> str:
        return RUN_STATE_NAMES.get(self.run_state, f"UNKNOWN_{self.run_state}")

    @property
    def deterministic_name(self) -> str:
        return VALIDATION_NAMES.get(self.validation_state, "UNKNOWN")


if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _kernel32.OpenFileMappingW.restype = wintypes.HANDLE
    _kernel32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
    _kernel32.MapViewOfFile.restype = ctypes.c_void_p
    _kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def _copy_struct(value: ctypes.Structure, kind: type[ctypes.Structure]):
    result = kind()
    ctypes.memmove(ctypes.addressof(result), ctypes.addressof(value), ctypes.sizeof(kind))
    return result


def validate_simple_patch(patch: SimpleStatePatch) -> None:
    for label, player in (("p1", patch.p1), ("p2", patch.p2)):
        for field in ("spirit", "maxSpirit"):
            value = getattr(player, field)
            if not -(1 << 15) <= value < (1 << 15):
                raise ValueError(f"{label}.{field} must fit signed 16-bit game storage")


class BridgeClient:
    _FILE_MAP_WRITE = 0x0002
    _FILE_MAP_READ = 0x0004

    def __init__(self, pid: int | None = None) -> None:
        if os.name != "nt":
            raise BridgeUnavailable("SokuRLBridge is only available on Windows")
        if pid is None:
            try:
                import psutil
                pids = [process.pid for process in psutil.process_iter(["name"])
                        if (process.info["name"] or "").casefold() == "th123.exe"]
            except Exception as error:
                raise BridgeUnavailable("a th123 PID is required") from error
            if len(pids) != 1:
                raise BridgeUnavailable(f"a th123 PID is required; found {len(pids)} instances")
            pid = pids[0]
        self.pid = pid
        mapping_name = MAPPING_NAME_FORMAT.format(pid)
        self._handle = _kernel32.OpenFileMappingW(
            self._FILE_MAP_READ | self._FILE_MAP_WRITE, False, mapping_name)
        if not self._handle:
            raise BridgeUnavailable(f"{mapping_name} is not available")
        self._view = _kernel32.MapViewOfFile(
            self._handle, self._FILE_MAP_READ | self._FILE_MAP_WRITE, 0, 0, MAPPING_SIZE)
        if not self._view:
            error = ctypes.get_last_error()
            _kernel32.CloseHandle(self._handle)
            self._handle = None
            raise BridgeUnavailable(f"MapViewOfFile failed with Windows error {error}")
        self._mapping_pointer = ctypes.cast(self._view, ctypes.POINTER(BridgeMapping))
        try:
            self._validate_abi()
        except Exception:
            self.close()
            raise

    @property
    def mapping(self) -> BridgeMapping:
        if self._mapping_pointer is None:
            raise BridgeUnavailable("bridge mapping is closed")
        return self._mapping_pointer.contents

    @property
    def block(self) -> ControlBlock:
        return self.mapping.control

    def _validate_abi(self) -> None:
        block = self.block
        expected = (CONTROL_MAGIC, CONTROL_VERSION, CONTROL_BLOCK_SIZE, MAPPING_SIZE)
        actual = (block.magic, block.version, block.structSize, block.mappingSize)
        if actual != expected:
            raise BridgeUnavailable(f"bridge ABI mismatch: DLL={actual}, Python={expected}")

    def close(self) -> None:
        self._mapping_pointer = None
        if getattr(self, "_view", None):
            _kernel32.UnmapViewOfFile(self._view)
            self._view = None
        if getattr(self, "_handle", None):
            _kernel32.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def snapshot(self) -> BridgeSnapshot:
        block = self.block
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            before = block.statusSeq
            if before & 1:
                time.sleep(0.001)
                continue
            latest = _copy_struct(block.latest, RawFrameState)
            values = (
                block.currentFrame, block.recordedFrames, block.lastVerifiedFrame,
                block.firstDivergentFrame, block.stepsRemaining,
            )
            after = block.statusSeq
            if before == after and not after & 1:
                break
            time.sleep(0.001)
        else:
            raise BridgeUnavailable("could not read a stable bridge snapshot")
        last_verified = None if values[2] == NO_FRAME else values[2]
        first_divergent = None if values[3] == NO_FRAME else values[3]
        return BridgeSnapshot(
            block.commandSeq, block.ackSeq, block.inputFramesRemaining, values[0],
            bool(block.connected), bool(block.inGameplay), block.resultCode, block.runState,
            bool(block.checkpointValid), block.validationState, bool(block.reconstructing),
            values[4], values[1], block.droppedFrames, last_verified, first_divergent, latest,
        )

    def _send(self, command: int, *, duration: int = 0, argument: int = 0) -> int:
        block = self.block
        block.durationFrames = duration
        block.commandArgument = argument
        block.commandType = command
        sequence = (block.commandSeq + 1) & 0xFFFFFFFF or 1
        block.commandSeq = sequence
        return sequence

    def reset_episode(self, seed):
        if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF:
            raise ValueError("reset seed must be in [0, 0xFFFFFFFF)")
        return self._send(COMMAND_RESET_EPISODE, argument=seed)

    def wait_for_reset(self, sequence, segment, seed, timeout):
        if timeout <= 0:
            raise ValueError("reset timeout must be positive")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if snapshot.ack_seq == sequence:
                if snapshot.result_code not in (2, 8):
                    raise RuntimeError(f"reset rejected with result {snapshot.result_code}")
                raw = snapshot.latest
                if (snapshot.in_gameplay and snapshot.checkpoint_valid and
                        snapshot.run_state_name == "PAUSED" and snapshot.game_frame == 0 and
                        raw.frameId == 0 and raw.segmentId == segment and raw.randomSeed == seed):
                    if raw.stateHash != calculate_state_hash(raw):
                        raise RuntimeError("reset frame-zero hash mismatch")
                    self.drain_frames()
                    return raw
            time.sleep(.001)
        raw = snapshot.latest
        raise TimeoutError(f"PID {self.pid}: reset did not reach a new paused frame zero; "
                           f"command={sequence} ack={snapshot.ack_seq} result={snapshot.result_code} "
                           f"gameplay={snapshot.in_gameplay} paused={snapshot.run_state_name} "
                           f"frame={snapshot.game_frame} segment={raw.segmentId}/{segment} "
                           f"seed={raw.randomSeed}/{seed} scene={raw.sceneId} "
                           f"checkpoint={snapshot.checkpoint_valid}")

    @staticmethod
    def _write_input(target: LogicalInput, values: tuple[int, ...] | LogicalInput) -> None:
        names = ("horizontalAxis", "verticalAxis", "a", "b", "c", "d", "changeCard", "spellcard")
        if isinstance(values, LogicalInput):
            values = tuple(getattr(values, name) for name in names)
        if len(values) != len(names):
            raise ValueError("logical input must contain eight values")
        for name, value in zip(names, values, strict=True):
            setattr(target, name, value)

    def send_action(self, action: str, frames: int) -> int:
        normalized = action.upper()
        if normalized not in ACTION_INPUTS:
            raise ValueError(f"unknown action: {action}")
        if not isinstance(frames, int) or not 1 <= frames <= MAX_DURATION_FRAMES:
            raise ValueError(f"frames must be an integer from 1 to {MAX_DURATION_FRAMES}")
        self._write_input(self.block.commandInput, ACTION_INPUTS[normalized])
        return self._send(COMMAND_INPUT, duration=frames)

    def release(self) -> int:
        self.block.commandInput = LogicalInput()
        return self._send(COMMAND_RELEASE)

    def run(self) -> int:
        return self._send(COMMAND_RUN)

    def pause(self) -> int:
        return self._send(COMMAND_PAUSE)

    def step(self, frames: int) -> int:
        if not isinstance(frames, int) or not 1 <= frames <= MAX_DURATION_FRAMES:
            raise ValueError(f"frames must be an integer from 1 to {MAX_DURATION_FRAMES}")
        return self._send(COMMAND_STEP_FRAMES, duration=frames)

    def establish_checkpoint(self, seed: int | None = None) -> int:
        if seed is not None and not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed must be a 32-bit unsigned integer")
        return self._send(
            COMMAND_ESTABLISH_CHECKPOINT,
            argument=NO_FRAME if seed is None else seed,
        )

    def step_with_inputs(
        self,
        p1: tuple[int, ...] | LogicalInput,
        p2: tuple[int, ...] | LogicalInput,
    ) -> int:
        self._write_input(self.block.commandInput, p1)
        self._write_input(self.block.commandInputP2, p2)
        return self._send(COMMAND_STEP_WITH_INPUTS, duration=1)

    def apply_simple_state(self, state: RawFrameState) -> int:
        patch = SimpleStatePatch()
        patch.timeElapsedRaw = state.timeElapsedRaw
        patch.activeWeather = state.activeWeather
        patch.displayedWeather = state.displayedWeather
        patch.weatherCounter = state.weatherCounter
        for target, source in ((patch.p1, state.p1), (patch.p2, state.p2)):
            for name in (
                "x", "y", "speedX", "speedY", "facing", "hp", "spirit",
                "maxSpirit", "cardGauge", "cardCount",
            ):
                setattr(target, name, getattr(source, name))
        return self.apply_simple_patch(patch)

    def apply_simple_patch(self, patch: SimpleStatePatch) -> int:
        validate_simple_patch(patch)
        ctypes.memmove(
            ctypes.addressof(self.block.commandPatch),
            ctypes.addressof(patch),
            ctypes.sizeof(SimpleStatePatch),
        )
        return self._send(COMMAND_APPLY_SIMPLE_STATE)

    def reconstruction_history(self) -> list[ReconstructionFrame]:
        snapshot = self.snapshot()
        count = snapshot.recorded_frames
        if count > INPUT_HISTORY_CAPACITY:
            raise BridgeUnavailable("reconstruction history exceeds shared-memory capacity")
        return [
            _copy_struct(self.mapping.history[index], ReconstructionFrame)
            for index in range(count)
        ]

    def goto_frame(self, frame: int) -> int:
        if not isinstance(frame, int) or frame < 0:
            raise ValueError("frame must be a non-negative integer")
        return self._send(COMMAND_GOTO_FRAME, argument=frame)

    def menu_confirm(self) -> int:
        return self._send(COMMAND_MENU_CONFIRM)

    def step_back(self) -> int:
        current = self.snapshot().game_frame
        if current == 0:
            raise ValueError("already at frame 0")
        return self.goto_frame(current - 1)

    def reset_ring(self) -> None:
        block = self.block
        block.ringReadSeq = block.ringWriteSeq
        block.droppedFrames = 0

    def drain_frames(self, limit: int | None = None) -> list[RawFrameState]:
        block = self.block
        write = block.ringWriteSeq
        read = block.ringReadSeq
        available = (write - read) & 0xFFFFFFFF
        if available > FRAME_RING_CAPACITY:
            raise BridgeUnavailable("ring sequence accounting is invalid")
        count = available if limit is None else min(available, limit)
        result = [
            _copy_struct(self.mapping.frames[(read + index) % FRAME_RING_CAPACITY], RawFrameState)
            for index in range(count)
        ]
        block.ringReadSeq = (read + count) & 0xFFFFFFFF
        return result

    def wait_for_ack(self, sequence: int, timeout: float = 2.0) -> BridgeSnapshot:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if snapshot.ack_seq == sequence:
                return snapshot
            time.sleep(0.01)
        return self.snapshot()


def calculate_state_hash(state: RawFrameState) -> int:
    fields = (
        "frameId", "sceneId", "battleMode", "battleSubMode", "stageId", "roundId",
        "timeElapsedRaw", "activeWeather", "displayedWeather", "weatherCounter", "randomSeed",
    )
    data = bytearray()
    for name in fields:
        field_type = dict(RawFrameState._fields_)[name]
        offset = getattr(RawFrameState, name).offset
        data.extend(ctypes.string_at(ctypes.addressof(state) + offset, ctypes.sizeof(field_type)))
    data.extend(ctypes.string_at(ctypes.addressof(state.p1), ctypes.sizeof(PlayerState)))
    data.extend(ctypes.string_at(ctypes.addressof(state.p2), ctypes.sizeof(PlayerState)))
    metadata_start = RawFrameState.p1ObjectCount.offset
    metadata_size = RawFrameState.p1Objects.offset - metadata_start
    data.extend(ctypes.string_at(ctypes.addressof(state) + metadata_start, metadata_size))
    data.extend(ctypes.string_at(ctypes.addressof(state.p1Objects), ctypes.sizeof(state.p1Objects)))
    data.extend(ctypes.string_at(ctypes.addressof(state.p2Objects), ctypes.sizeof(state.p2Objects)))
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value

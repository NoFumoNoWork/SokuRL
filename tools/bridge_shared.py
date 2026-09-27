from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from dataclasses import dataclass

MAPPING_NAME = r"Local\SokuRLBridge"
CONTROL_MAGIC = 0x554B4F53
CONTROL_VERSION = 1
MAX_DURATION_FRAMES = 10_000

COMMAND_INPUT = 1
COMMAND_RELEASE = 2

RESULT_NAMES = {
    0: "IDLE",
    1: "ACCEPTED",
    2: "COMPLETE",
    3: "RELEASED",
    4: "NOT_IN_GAMEPLAY",
    5: "INVALID_COMMAND",
}

ACTION_INPUTS = {
    "NEUTRAL": (0, 0, 0, 0, 0, 0),
    "LEFT": (-1, 0, 0, 0, 0, 0),
    "RIGHT": (1, 0, 0, 0, 0, 0),
    "UP": (0, -1, 0, 0, 0, 0),
    "DOWN": (0, 1, 0, 0, 0, 0),
    "UP_LEFT": (-1, -1, 0, 0, 0, 0),
    "UP_RIGHT": (1, -1, 0, 0, 0, 0),
    "DOWN_LEFT": (-1, 1, 0, 0, 0, 0),
    "DOWN_RIGHT": (1, 1, 0, 0, 0, 0),
    "A": (0, 0, 1, 0, 0, 0),
    "B": (0, 0, 0, 1, 0, 0),
    "C": (0, 0, 0, 0, 1, 0),
    "D": (0, 0, 0, 0, 0, 1),
    "LEFT_A": (-1, 0, 1, 0, 0, 0),
    "RIGHT_A": (1, 0, 1, 0, 0, 0),
    "DOWN_A": (0, 1, 1, 0, 0, 0),
    "UP_A": (0, -1, 1, 0, 0, 0),
    "LEFT_B": (-1, 0, 0, 1, 0, 0),
    "RIGHT_B": (1, 0, 0, 1, 0, 0),
    "DOWN_B": (0, 1, 0, 1, 0, 0),
    "UP_B": (0, -1, 0, 1, 0, 0),
    "LEFT_C": (-1, 0, 0, 0, 1, 0),
    "RIGHT_C": (1, 0, 0, 0, 1, 0),
    "DOWN_C": (0, 1, 0, 0, 1, 0),
    "UP_C": (0, -1, 0, 0, 1, 0),
}


class ControlBlock(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("structSize", ctypes.c_uint32),
        ("commandSeq", ctypes.c_uint32),
        ("ackSeq", ctypes.c_uint32),
        ("commandType", ctypes.c_uint32),
        ("horizontalAxis", ctypes.c_int32),
        ("verticalAxis", ctypes.c_int32),
        ("a", ctypes.c_uint32),
        ("b", ctypes.c_uint32),
        ("c", ctypes.c_uint32),
        ("d", ctypes.c_uint32),
        ("durationFrames", ctypes.c_uint32),
        ("framesRemaining", ctypes.c_uint32),
        ("gameFrame", ctypes.c_uint64),
        ("connected", ctypes.c_uint32),
        ("inGameplay", ctypes.c_uint32),
        ("resultCode", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


CONTROL_BLOCK_SIZE = ctypes.sizeof(ControlBlock)
assert CONTROL_BLOCK_SIZE == 80
assert ControlBlock.commandSeq.offset == 12
assert ControlBlock.horizontalAxis.offset == 24
assert ControlBlock.durationFrames.offset == 48
assert ControlBlock.gameFrame.offset == 56
assert ControlBlock.resultCode.offset == 72


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

    @property
    def result_name(self) -> str:
        return RESULT_NAMES.get(self.result_code, f"UNKNOWN_{self.result_code}")


if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _kernel32.OpenFileMappingW.restype = wintypes.HANDLE
    _kernel32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
    _kernel32.MapViewOfFile.restype = ctypes.c_void_p
    _kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    _kernel32.UnmapViewOfFile.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL


class BridgeClient:
    _FILE_MAP_WRITE = 0x0002
    _FILE_MAP_READ = 0x0004

    def __init__(self) -> None:
        if os.name != "nt":
            raise BridgeUnavailable("SokuRLBridge is only available on Windows")

        self._handle = _kernel32.OpenFileMappingW(
            self._FILE_MAP_READ | self._FILE_MAP_WRITE,
            False,
            MAPPING_NAME,
        )
        if not self._handle:
            raise BridgeUnavailable("Local\\SokuRLBridge is not available; start th123 with the module loaded")

        self._view = _kernel32.MapViewOfFile(
            self._handle,
            self._FILE_MAP_READ | self._FILE_MAP_WRITE,
            0,
            0,
            CONTROL_BLOCK_SIZE,
        )
        if not self._view:
            error = ctypes.get_last_error()
            _kernel32.CloseHandle(self._handle)
            self._handle = None
            raise BridgeUnavailable(f"MapViewOfFile failed with Windows error {error}")

        self._block_pointer = ctypes.cast(self._view, ctypes.POINTER(ControlBlock))
        try:
            self._validate_abi()
        except Exception:
            self.close()
            raise

    @property
    def block(self) -> ControlBlock:
        if self._block_pointer is None:
            raise BridgeUnavailable("bridge mapping is closed")
        return self._block_pointer.contents

    def _validate_abi(self) -> None:
        block = self.block
        if block.magic != CONTROL_MAGIC:
            raise BridgeUnavailable(f"bridge magic mismatch: 0x{block.magic:08X}")
        if block.version != CONTROL_VERSION:
            raise BridgeUnavailable(f"bridge version mismatch: {block.version}")
        if block.structSize != CONTROL_BLOCK_SIZE:
            raise BridgeUnavailable(
                f"bridge size mismatch: DLL={block.structSize}, Python={CONTROL_BLOCK_SIZE}"
            )

    def close(self) -> None:
        if getattr(self, "_block_pointer", None) is not None:
            self._block_pointer = None
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
        game_frame = block.gameFrame
        for _ in range(2):
            repeated = block.gameFrame
            if repeated == game_frame:
                break
            game_frame = repeated
        return BridgeSnapshot(
            command_seq=block.commandSeq,
            ack_seq=block.ackSeq,
            frames_remaining=block.framesRemaining,
            game_frame=game_frame,
            connected=bool(block.connected),
            in_gameplay=bool(block.inGameplay),
            result_code=block.resultCode,
        )

    def send_action(self, action: str, frames: int) -> int:
        normalized = action.upper()
        if normalized not in ACTION_INPUTS:
            raise ValueError(f"unknown action: {action}")
        if not isinstance(frames, int) or not 1 <= frames <= MAX_DURATION_FRAMES:
            raise ValueError(f"frames must be an integer from 1 to {MAX_DURATION_FRAMES}")

        block = self.block
        horizontal, vertical, a, b, c, d = ACTION_INPUTS[normalized]
        block.horizontalAxis = horizontal
        block.verticalAxis = vertical
        block.a = a
        block.b = b
        block.c = c
        block.d = d
        block.durationFrames = frames
        block.commandType = COMMAND_INPUT
        sequence = (block.commandSeq + 1) & 0xFFFFFFFF
        if sequence == 0:
            sequence = 1
        block.commandSeq = sequence
        return sequence

    def release(self) -> int:
        block = self.block
        block.horizontalAxis = 0
        block.verticalAxis = 0
        block.a = 0
        block.b = 0
        block.c = 0
        block.d = 0
        block.durationFrames = 0
        block.commandType = COMMAND_RELEASE
        sequence = (block.commandSeq + 1) & 0xFFFFFFFF
        if sequence == 0:
            sequence = 1
        block.commandSeq = sequence
        return sequence

    def wait_for_ack(self, sequence: int, timeout: float = 2.0) -> BridgeSnapshot:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if snapshot.ack_seq == sequence:
                return snapshot
            time.sleep(0.01)
        return self.snapshot()

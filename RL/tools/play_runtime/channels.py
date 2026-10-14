"""Exchange bounded snapshots and expiring actions without controlling game pace."""
import ctypes as C
from dataclasses import dataclass
import struct
import threading

import bridge_shared
from game_runtime.snapshot_memory import SnapshotMemory
from soku_rl.env.observation.render_state import RENDER_STATE_SIZE, RenderSnapshot
from soku_rl.play.match import MatchState


HISTORY_HEADER = struct.Struct("<4IQII")
FRAME_HEADER = struct.Struct("<IIQIIQ")
RAW_SIZE = C.sizeof(bridge_shared.RawFrameState)
REGION_COUNT, BYTE_COUNT, CAPACITY = 40000, 4 * 1024 * 1024, 16
MEMORY_OFFSET = FRAME_HEADER.size + RAW_SIZE + RENDER_STATE_SIZE
DATA_OFFSET = MEMORY_OFFSET + 12 + REGION_COUNT * 12
FRAME_SIZE = DATA_OFFSET + BYTE_COUNT
HISTORY_SIZE = HISTORY_HEADER.size + CAPACITY * FRAME_SIZE
COMMAND = struct.Struct("<4I3Q8i")
STATUS = struct.Struct("<5IQIQI8i")
INPUT_SIZE = 168
RESULTS = ("idle", "accepted", "wrong_context", "invalid", "stale", "expired", "queue_full")


class _Mapping:
    def __init__(self, pid, name, size, write):
        if type(pid) is not int or pid <= 0:
            raise ValueError("a positive owned game PID is required")
        self.kernel = bridge_shared._kernel32
        access = 6 if write else 4
        self.handle = self.kernel.OpenFileMappingW(access, False, rf"Local\SokuRL{name}_{pid}")
        if not self.handle:
            raise OSError(C.get_last_error(), f"{name} mapping is unavailable")
        self.view = self.kernel.MapViewOfFile(self.handle, access, 0, 0, size)
        if not self.view:
            error = C.get_last_error()
            self.kernel.CloseHandle(self.handle)
            raise OSError(error, f"cannot map {name}")

    def close(self):
        if self.view:
            self.kernel.UnmapViewOfFile(self.view)
            self.view = 0
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = 0


@dataclass(frozen=True)
class CapturedFrame:
    match: MatchState
    raw: object
    render: RenderSnapshot
    memory: SnapshotMemory
    capture_seconds: float


@dataclass(frozen=True)
class SnapshotFrame:
    raw: object
    render: RenderSnapshot
    memory: SnapshotMemory
    capture_seconds: float
    scores: tuple


class SnapshotHistory(_Mapping):
    def __init__(self, pid):
        super().__init__(pid, "RealtimeHistory", HISTORY_SIZE, False)
        header = HISTORY_HEADER.unpack(C.string_at(self.view, HISTORY_HEADER.size))
        if header[:4] != (0x48524B53, 1, HISTORY_SIZE, CAPACITY) or header[4] <= 0:
            self.close()
            raise ValueError("unsupported realtime history ABI")
        self.frequency = header[4]

    def read_after(self, cursor):
        if not self.view or type(cursor) is not int or not 0 <= cursor < 2**32:
            raise ValueError("read requires an open history and uint32 cursor")
        _, _, _, _, _, written, alive = HISTORY_HEADER.unpack(C.string_at(self.view, HISTORY_HEADER.size))
        if not alive:
            raise EOFError("game observation producer closed")
        count = (written - cursor) & 0xFFFFFFFF
        if count > CAPACITY:
            raise BufferError(f"AI fell behind; {count-CAPACITY} observation frames were overwritten")
        frames = []
        for i in range(count):
            index = (cursor + i) & 0xFFFFFFFF
            address = self.view + HISTORY_HEADER.size + (index % CAPACITY) * FRAME_SIZE
            before = C.c_uint32.from_address(address).value
            if before & 1:
                break
            prefix = C.string_at(address, MEMORY_OFFSET + 12)
            regions, used, error = struct.unpack_from("<3I", prefix, MEMORY_OFFSET)
            if regions > REGION_COUNT or used > BYTE_COUNT:
                raise RuntimeError("invalid native snapshot bounds")
            descriptors = C.string_at(address + MEMORY_OFFSET + 12, regions * 12)
            data = C.string_at(address + DATA_OFFSET, used)
            after = C.c_uint32.from_address(address).value
            now = C.c_uint32.from_address(self.view + 24).value
            if (now-index) & 0xFFFFFFFF > CAPACITY:
                raise BufferError("AI observation was overwritten during copying")
            if before != after or before != struct.unpack_from("<I", prefix)[0]:
                break
            if error:
                raise RuntimeError(f"native complete-state capture failed with code {error}")
            _, match, frame, left, right, ticks = FRAME_HEADER.unpack_from(prefix)
            raw = bridge_shared.RawFrameState.from_buffer_copy(prefix, FRAME_HEADER.size)
            if (raw.segmentId, raw.frameId) != (match, frame):
                raise RuntimeError("inconsistent native frame identity")
            render = RenderSnapshot.decode(prefix[FRAME_HEADER.size + RAW_SIZE:MEMORY_OFFSET])
            memory = SnapshotMemory(struct.iter_unpack("<3I", descriptors), data)
            frames.append(SnapshotFrame(raw, render, memory, ticks / self.frequency, (left, right)))
        return (cursor + len(frames)) & 0xFFFFFFFF, tuple(frames)


class RealtimeHistory(SnapshotHistory):
    def read_after(self, cursor):
        end, snapshots = super().read_after(cursor)
        frames = []
        for snapshot in snapshots:
            raw = snapshot.raw
            state = MatchState(raw.segmentId, raw.roundId, raw.frameId, snapshot.scores,
                               (raw.p1.hp, raw.p2.hp), "battle")
            frames.append(CapturedFrame(state, raw, snapshot.render, snapshot.memory, snapshot.capture_seconds))
        return end, tuple(frames)

    def read_latest(self, cursor):
        """Prefer current play over replaying a backlog; keep strict reads separate."""
        if not self.view or type(cursor) is not int or not 0 <= cursor < 2**32:
            raise ValueError("read requires an open history and uint32 cursor")
        written = C.c_uint32.from_address(self.view + 24).value
        count = (written - cursor) & 0xFFFFFFFF
        start = (written - 1) & 0xFFFFFFFF if count else cursor
        try:
            end, frames = self.read_after(start)
        except BufferError:
            # The writer lapped this copy. Retry the newest slot on the next poll.
            return cursor, (), 0
        if not frames:
            return cursor, (), 0
        return end, frames[-1:], ((end - cursor) & 0xFFFFFFFF) - 1


class RealtimeInput(_Mapping):
    def __init__(self, pid, seat):
        if type(seat) is not int or seat not in (0, 1):
            raise ValueError("AI seat must be zero or one")
        super().__init__(pid, "RealtimeInput", INPUT_SIZE, True)
        if struct.unpack("<4I", C.string_at(self.view, 16)) != (0x54524B53, 1, INPUT_SIZE, seat):
            self.close()
            raise ValueError("realtime input ABI or AI seat differs")
        self.seat, self.sequence = seat, 0
        self.writer = threading.get_ident()

    def status(self):
        if not self.view:
            raise RuntimeError("input channel is closed")
        before = C.c_uint32.from_address(self.view + 92).value
        values = STATUS.unpack(C.string_at(self.view + 92, STATUS.size))
        after = C.c_uint32.from_address(self.view + 92).value
        if before != after or before != values[0] or before & 1:
            return None
        _, acknowledged, result, match, round_id, frame, applied, at, pending, *held = values
        if result >= len(RESULTS):
            raise RuntimeError("unknown realtime input result")
        return dict(acknowledged=acknowledged, result=RESULTS[result], match=match, round=round_id,
                    frame=frame, applied=applied, applied_at=at, pending=pending, held=tuple(held))

    def submit(self, state, keys, latency, lifetime):
        if threading.get_ident() != self.writer:
            raise RuntimeError("realtime input has exactly one writer thread")
        if (len(keys) != 8 or any(type(k) is not int for k in keys) or
                any(k not in (-1, 0, 1) for k in keys[:2]) or any(k not in (0, 1) for k in keys[2:])):
            raise ValueError("eight valid logical axes and buttons are required")
        if (type(latency) is not int or not 0 <= latency <= 10000 or
                type(lifetime) is not int or not 1 <= lifetime <= 10000):
            raise ValueError("invalid action latency or lifetime")
        status = self.status()
        if status is None or status["acknowledged"] != self.sequence:
            return False
        sequence = (self.sequence + 1) & 0xFFFFFFFF or 1
        target = state.frame + latency
        data = COMMAND.pack(sequence, state.match, state.round, self.seat,
                            state.frame, target, target + lifetime, *keys)
        # One producer, aligned uint32 stores, x86/x64 Windows/Wine. Store order
        # is preserved on these architectures; no read-modify-write API is needed.
        # InterlockedIncrement is an intrinsic, not a portable kernel32 export.
        guard = C.c_uint32.from_address(self.view + 16)
        guard.value = (guard.value + 1) & 0xFFFFFFFF
        C.memmove(self.view + 20, data, COMMAND.size)
        guard.value = (guard.value + 1) & 0xFFFFFFFF
        self.sequence = sequence
        return True

    def submit_current(self, state, keys, latency, lifetime):
        """Preserve minimum reaction delay but give late actions a future deadline."""
        target = state.frame + latency
        status = self.status()
        valid = status is not None and (status["match"], status["round"]) == (state.match, state.round)
        if valid:
            target = max(target, status["frame"] + 1)
        delay = target - state.frame
        submitted = valid and delay <= 10000 and self.submit(state, keys, delay, lifetime)
        return {"submitted": submitted, "request": self.sequence,
                "target": target, "expires": target + lifetime}

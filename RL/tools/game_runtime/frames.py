"""Read and acknowledge native frame records, including the paused initial state."""
import ctypes
import math
import time

from bridge_shared import BridgeUnavailable, FRAME_RING_CAPACITY, RawFrameState, calculate_state_hash


FRAME_SIZE = ctypes.sizeof(RawFrameState)


def drain_frames_into(client, buffer):
    """Copy queued frames in order before releasing their slots to the producer."""
    block = client.block
    write, read = block.ringWriteSeq, block.ringReadSeq
    available = (write - read) & 0xFFFFFFFF
    if available > FRAME_RING_CAPACITY:
        raise BridgeUnavailable("ring sequence accounting is invalid")
    if not isinstance(buffer, ctypes.Array) or ctypes.sizeof(buffer) < available * FRAME_SIZE:
        raise ValueError("frame buffer cannot hold the queued records")
    if not available:
        return 0
    first_index = read % FRAME_RING_CAPACITY
    first_count = min(available, FRAME_RING_CAPACITY - first_index)
    ctypes.memmove(ctypes.addressof(buffer),
                   ctypes.addressof(client.mapping.frames[first_index]), first_count * FRAME_SIZE)
    remaining = available - first_count
    if remaining:
        ctypes.memmove(ctypes.addressof(buffer) + first_count * FRAME_SIZE,
                       ctypes.addressof(client.mapping.frames[0]), remaining * FRAME_SIZE)
    block.ringReadSeq = write
    return available


def wait_for_frame_zero(client, pid, timeout):
    """Return an owned copy only after the initial paused state passes its hash check."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("frame-zero timeout must be positive and finite")
    deadline = time.monotonic() + timeout
    while True:
        snapshot = client.snapshot()
        if (snapshot.in_gameplay and snapshot.checkpoint_valid and snapshot.game_frame == 0
                and snapshot.run_state_name == "PAUSED"):
            state = RawFrameState.from_buffer_copy(snapshot.latest)
            if state.stateHash != calculate_state_hash(state):
                raise RuntimeError(f"PID {pid}: native/Python frame-zero hash mismatch")
            client.drain_frames()
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"PID {pid}: timed out waiting for paused VS frame zero; "
                               f"gameplay={snapshot.in_gameplay} checkpoint={snapshot.checkpoint_valid} "
                               f"frame={snapshot.game_frame} run_state={snapshot.run_state_name} "
                               f"scene={snapshot.latest.sceneId} result={snapshot.result_code}")
        time.sleep(0.005)

import ctypes
import random
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bridge_shared as b


@pytest.mark.parametrize("kind", list(b.RAW_FRAME_TYPES_BY_SIZE.values()))
def test_exact_native_hash(kind):
    if b.HASH_BACKEND != "cpp-fnv64":
        pytest.skip("optional x64 host helper not built")
    for seed in range(3):
        state = kind.from_buffer_copy(random.Random(seed).randbytes(ctypes.sizeof(kind)))
        assert b.calculate_state_hash(state, backend="cpp") == b.calculate_state_hash(state, backend="python")


def client_with_ring():
    mapping = b.BridgeMapping()
    client = b.BridgeClient.__new__(b.BridgeClient)
    client._mapping_pointer = ctypes.pointer(mapping)
    return client, mapping


def test_bulk_ring_owned_wrap_and_limit():
    client, mapping = client_with_ring()
    mapping.control.ringReadSeq = b.FRAME_RING_CAPACITY - 1
    mapping.control.ringWriteSeq = b.FRAME_RING_CAPACITY + 2
    for index, frame in [(b.FRAME_RING_CAPACITY - 1, 10), (0, 11), (1, 12)]:
        mapping.frames[index].frameId = frame
    payload = client.drain_frame_bytes(2)
    size = ctypes.sizeof(b.RawFrameState)
    assert [b.RawFrameState.from_buffer_copy(payload, i*size).frameId for i in range(2)] == [10, 11]
    mapping.frames[0].frameId = 99
    assert b.RawFrameState.from_buffer_copy(payload, size).frameId == 11
    assert len(client.drain_frame_bytes()) == size
    assert client.drain_frame_bytes() == bytearray()


def test_bulk_ring_rejects_invalid_accounting():
    client, mapping = client_with_ring()
    mapping.control.ringWriteSeq = b.FRAME_RING_CAPACITY + 1
    with pytest.raises(Exception):
        client.drain_frame_bytes()


def test_reset_polling_only_snapshots_at_completion(monkeypatch):
    client, mapping = client_with_ring()
    raw = b.RawFrameState()
    raw.segmentId = 3
    raw.randomSeed = 123
    raw.stateHash = b.calculate_state_hash(raw)
    calls = []
    sleeps = []
    def snapshot():
        calls.append(1)
        return SimpleNamespace(latest=raw,in_gameplay=True,checkpoint_valid=True,
                               run_state_name="PAUSED",game_frame=0)
    def sleep(_):
        sleeps.append(1)
        if len(sleeps) == 3:
            block = mapping.control
            block.ackSeq = 7
            block.resultCode = 2
            block.inGameplay = 1
            block.checkpointValid = 1
            block.runState = 1
    client.snapshot = snapshot
    client.drain_frames = lambda: []
    monkeypatch.setattr(b.time,"sleep",sleep)
    assert client.wait_for_reset(7,3,123,1) is raw
    assert len(sleeps) == 3
    assert len(calls) == 1

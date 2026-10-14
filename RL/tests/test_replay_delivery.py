"""A failed game request must retain any replay recovered during cleanup."""
import hashlib
import json

import pytest

from soku_rl.env.worker_pipe import WorkerBackend


def test_error_response_saves_recovered_episode_before_raising(tmp_path):
    backend = WorkerBackend.__new__(WorkerBackend)
    backend.replay_directory = tmp_path / "replays"
    backend.replay_number = 0
    data = b"recovered original replay bytes"
    response = {"ok": False, "error": "invalid logical input", "replays": [
        {"slot": 2, "seed": 1732, "frames": 600, "reason": "close", "data": data}]}
    with pytest.raises(RuntimeError, match="invalid logical input"):
        backend._response_value(response)
    replay, = backend.replay_directory.glob("*.rep")
    assert replay.read_bytes() == data
    metadata = json.loads(replay.with_suffix(".json").read_text())
    assert metadata == {"slot": 2, "seed": 1732, "frames": 600, "reason": "close",
                        "sha256": hashlib.sha256(data).hexdigest(), "replay": replay.name}

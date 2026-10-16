"""An obsolete native recorder must not silently export unprotected episodes."""
from types import SimpleNamespace

import pytest

from game_runtime.recording import EpisodeRecording


def test_export_requires_boundary_capability_and_closes_memory_on_failure():
    closed = []
    recording = EpisodeRecording.__new__(EpisodeRecording)
    recording.memory = SimpleNamespace(close=lambda:closed.append(True))
    with pytest.raises(RuntimeError,match="input-boundary protection"):
        recording.finish(SimpleNamespace(original_replay_recording_enabled=False),'close')
    assert closed == [True]

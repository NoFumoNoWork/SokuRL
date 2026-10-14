import ctypes
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bridge_shared import CONTROL_VERSION, COMMAND_CONFIGURE_TRACE, RawFrameState
from game_runtime.batch import SokuGameBatch
from game_runtime.offline_snapshot import offline_snapshot_requested
from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.env.observation.render_state import CAMERA, ENTITY, COUNTS, RenderSnapshot
from soku_rl.env.worker_pipe import WorkerConnection


def test_current_platform_abi_and_trace_command_preserved():
    assert CONTROL_VERSION == 12 and COMMAND_CONFIGURE_TRACE == 13
    assert ctypes.sizeof(RawFrameState) == 12436


@pytest.mark.parametrize("capacity", (64, 1024))
def test_render_decoder_negotiates_native_capacity_without_padding_missing_data(capacity):
    data = bytearray(CAMERA.size + ENTITY.size * (2 + 2 * capacity) + COUNTS.size)
    CAMERA.pack_into(data, 0, 0, 480, 1, 21)
    ENTITY.pack_into(data, CAMERA.size + ENTITY.size * (2 + capacity), 123, 45, 1, -1, 1)
    COUNTS.pack_into(data, len(data) - COUNTS.size, 0, 1, 0)
    result = RenderSnapshot.decode(bytes(data), capacity)
    assert result.objects[1][0].x == 123
    with pytest.raises(ValueError, match="length"):
        RenderSnapshot.decode(bytes(data), 1024 if capacity == 64 else 64)


def test_fresh_match_configuration_is_scoped_before_launch():
    backend = SokuGameBatch(40)
    backend.fresh_reset = True
    backend._check_match(MatchConfig(PlayerSetup(1, 0, 0), PlayerSetup(0, 0, 0)))
    for setup in (PlayerSetup(2, 0, 0), PlayerSetup(0, 1, 0), PlayerSetup(0, 0, 1)):
        with pytest.raises(NotImplementedError):
            backend._check_match(MatchConfig(setup, PlayerSetup(1, 0, 0)))


def test_worker_rejects_unvalidated_reset_combinations_before_popen(tmp_path):
    with pytest.raises(ValueError, match="fresh reset requires"):
        WorkerConnection(["not-launched"], ".", tmp_path / "worker.log", 60, 40,
                         True, "th123_jp", fresh_reset=True)
    assert not (tmp_path / "worker.log").exists()


def test_fork_snapshot_requests_fail_explicitly(monkeypatch):
    monkeypatch.setenv("SOKURL_OFFLINE_SNAPSHOT", "1")
    with pytest.raises(NotImplementedError, match="ABI12"):
        offline_snapshot_requested()

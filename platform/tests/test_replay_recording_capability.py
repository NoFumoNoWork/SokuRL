"""Replay startup protection has an opt-in bit independent of fast reset."""
from types import SimpleNamespace

import pytest

from sokurl_platform.bridge_shared import BridgeClient


@pytest.mark.parametrize("bits,enabled", [(0,False),(1,False),(2,True),(3,True)])
def test_original_recording_protection_is_explicit(bits, enabled):
    client = BridgeClient.__new__(BridgeClient)
    client._mapping_pointer = SimpleNamespace(contents=SimpleNamespace(
        control=SimpleNamespace(traceReserved=bits)))
    assert client.original_replay_recording_enabled is enabled
    assert client.fresh_reset_enabled is bool(bits & 1)

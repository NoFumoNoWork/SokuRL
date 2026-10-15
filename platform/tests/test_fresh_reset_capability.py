import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bridge_shared as bridge


@pytest.mark.parametrize("bits, expected", [(0, False), (1, True), (2, False), (3, True)])
def test_fresh_reset_capability_is_explicit(bits, expected):
    client = bridge.BridgeClient.__new__(bridge.BridgeClient)
    client._mapping_pointer = SimpleNamespace(contents=SimpleNamespace(
        control=SimpleNamespace(traceReserved=bits)))
    assert client.fresh_reset_enabled is expected

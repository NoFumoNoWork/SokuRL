"""Explicit original CPU null-manager path, without changing human key reads."""
import pytest
from sokurl_platform.native_reader import PrivilegedReader


def test_human_key_manager_remains_the_original_source():
    reader = PrivilegedReader.__new__(PrivilegedReader)
    calls = []
    def read(address, kind):
        calls.append((address,kind))
        return 0x2000 if address == 0x1000 else tuple(range(8))
    reader.value = read
    assert reader.player_keys(0x3000,0x1000) == tuple(range(8))
    assert calls == [(0x1000,'I'),(0x2038,'8i')]


def test_cpu_player_keys_require_actual_vscom_mode():
    reader = PrivilegedReader.__new__(PrivilegedReader)
    calls = []
    def read(address, kind):
        calls.append((address,kind))
        return 2 if address == 0x898690 else (1,0,2,0,0,0,0,0)
    reader.value = read
    assert reader.player_keys(0x3000,0) == (1,0,2,0,0,0,0,0)
    assert calls == [(0x898690,'B'),(0x3754,'8i')]
    reader.value = lambda *_: 3
    with pytest.raises(RuntimeError,match='outside native VSCOM'):
        reader.player_keys(0x3000,0)

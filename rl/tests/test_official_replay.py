"""Original replay files must survive a byte-identical read/write round trip."""
from pathlib import Path
import os
import struct

import pytest

from soku_rl.replay import Replay, ReplayMatch, ReplayPlayer


def example():
    players = tuple(ReplayPlayer(character, seat, tuple(range(20)), seat, 0, 1)
                    for seat, character in enumerate((1, 0)))
    match = ReplayMatch(players, 3, 0, 0, 1732, (0, 2, 0x300, 0, 0x3528))
    return Replay(0xD2, bytes((0, 0, 9, 30, 0, 0, 0, 1, 3, 0)), (match,))


def test_original_game_replays_round_trip_without_changing_one_byte():
    from sokurl_platform.resources import game_directory
    files = list((game_directory() / 'replay').glob('*.rep'))
    if not files:
        pytest.skip("external original game replays are unavailable")
    for path in files:
        data = path.read_bytes()
        decoded = Replay.decode(data)
        assert decoded.encode() == data, path.name


def test_replay_round_trip_preserves_metadata_and_all_input_bits():
    replay = example()
    assert Replay.decode(replay.encode()) == replay


@pytest.mark.parametrize("change", ("version", "count", "truncated", "extra", "deck"))
def test_malformed_replay_is_rejected(change):
    data = bytearray(example().encode())
    if change == "version":
        data[0] = 0
    elif change == "count":
        data[11] = 2
    elif change == "truncated":
        data.pop()
    elif change == "extra":
        data.extend(b"extra")
    else:
        struct.pack_into("<I", data, 16, 21)
    with pytest.raises(ValueError):
        Replay.decode(bytes(data))

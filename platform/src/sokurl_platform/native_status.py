"""Known original match/replay fields, read through an owned memory reader."""
import struct
from .native_layout import FIGHTER_FIELDS


def match_scores(memory):
    battle, = struct.unpack('<I', memory.read(0x8985E4, 4))
    players = struct.unpack('<2I', memory.read(battle + 0xC, 8))
    offset, kind = FIGHTER_FIELDS['win_count']
    return tuple(struct.unpack('<' + kind, memory.read(player + offset, 1))[0] for player in players)


def remaining_replay_inputs(memory):
    record, = struct.unpack('<I', memory.read(0x898718 + 0x104, 4))
    if not record:
        raise RuntimeError('original replay input record is unavailable')
    remaining, = struct.unpack('<I', memory.read(record + 0x4C, 4))
    return remaining

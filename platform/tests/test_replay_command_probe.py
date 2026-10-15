import sys
from pathlib import Path
from types import SimpleNamespace
import struct

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from probe_replay_commands import logical, rebuild_commands, noncard_diff, candidates
from bridge_shared import RawFrameState


def test_counter_projection_keeps_absolute_direction_and_simultaneous_buttons():
    assert logical((-12, 45, 7, 2, 0, 0, 3, 1)) == (-1, 1, 1, 1, 0, 0, 1, 1)


def test_poll_rebuild_keeps_incomplete_pair_and_unknown_bits_without_invented_tick():
    values = {0x10: (0, 0, 1, 0, 0, 0, 0, 0),
              0x28: (1, 0, 0, 1, 0, 0, 0, 0),
              0x410: (0, 0, 1, 0, 0, 0, 0, 0)}
    names = ('x', 'y', 'A', 'B', 'C', 'D', 'E', 'F')
    module = SimpleNamespace(decode_keys=lambda word: dict(zip(names, values[word])),
        encode=lambda data: struct.pack('<3H', *data['rounds'][0]['input_words']))
    decoded = dict(input_player_mask=3, rounds=[dict(round_index=0, input_words=[0x10, 0x28, 0x410])])
    output, programs = rebuild_commands(module, decoded)
    assert output == struct.pack('<3H', 0x10, 0x28, 0x410)
    assert [p['samples'] for p in programs] == [2, 1]
    assert all(p['tick_semantics'] == 'NATIVE_INPUT_POLL_NOT_BATTLE_FRAME' for p in programs)
    neutral, _ = rebuild_commands(module, decoded, neutral=True)
    assert neutral == struct.pack('<3H', 0, 0, 0x400)


def test_noncard_projection_does_not_hide_damage_or_projectile_divergence():
    expected, actual = RawFrameState(), RawFrameState()
    actual.p1.cardCount = 3
    actual.battleSubMode = 1
    assert noncard_diff(expected, actual) == []
    actual.p1.hp = 42
    actual.p2Objects[2].x = 100
    differences = noncard_diff(expected, actual)
    assert {d['field'] for d in differences} == {'p1.hp', 'p2Objects[2].x'}


def test_combo_reset_inside_resolver_splits_contact_windows():
    events = []
    for root, frame in ((1, 1), (2, 3)):
        events.append(dict(kind='ACTION_TRANSITION', accepted=True, attacker_seat=0,
                           attack_instance=root, action_after=300, frame=frame))
    for frame, root, before, after in ((1, 1, 0, 1), (2, 1, 1, 2),
                                       (3, 2, 2, 1), (4, 2, 1, 2)):
        events.append(dict(kind='HIT_RESOLUTION', frame=frame, attacker_seat=0,
            defender_seat=1, attack_instance=root, source_action=300,
            hp_before=1000, hp_after=900, combo_count_before=before, combo_count_after=after))
    events.sort(key=lambda event: event['frame'])
    inputs = [dict(logical=[(0,)*8, (0,)*8]) for _ in range(4)]
    groups = candidates(events, inputs, [RawFrameState() for _ in range(5)])
    assert [(group['first_hit'], group['last_hit'], group['hits']) for group in groups] == [(1, 2, 2), (3, 4, 2)]

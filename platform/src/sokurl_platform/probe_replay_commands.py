"""Compare observed replay-frame inputs with the shared command interpreter.

Replay parsing and native playback reuse the existing Platform tools. This is
an evidence experiment, not a replay-poll-index-to-frame assumption.
"""
import argparse
import ctypes
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

from sokurl_platform.combat_capture import CombatCaptureClient
from sokurl_platform.command_input import CommandPlayer, compile_input_frames
from sokurl_platform.frame_validation import InputPair, input_tuple, copy_state, state_diff
from sokurl_platform.replay_validation import launch_replay_checkpoint
from sokurl_platform.bridge_shared import RawFrameState
from sokurl_platform.bridge_shared import BridgeClient, BridgeUnavailable
from sokurl_platform.frame_validation import PracticeInstance
from sokurl_platform import sokurl


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def logical(values):
    return tuple((x > 0) - (x < 0) for x in values[:2]) + tuple(int(x > 0) for x in values[2:])


def battle_diff(expected, actual, live_submode=False):
    # Mode/input transport metadata is not the battle state projection. Native
    # frame-aligned input counters are separately compared in the sidecar.
    excluded = {'stateHash', 'segmentId'} | ({'battleSubMode'} if live_submode else set())
    return [d for d in state_diff(expected, actual)
            if d['field'] not in excluded
            and not d['field'].startswith(('p1.input.', 'p2.input.'))]


def capture(replay, frames, folder):
    os.environ['SOKURL_NATIVE_INPUT_COUNTERS'] = '0'
    instance = launch_replay_checkpoint(replay)
    events = CombatCaptureClient(instance.pid)
    states = [copy_state(instance.state)]
    contacts = []
    inputs = []
    try:
        events.drain()
        while len(inputs) < frames:
            count = min(64, frames - len(inputs))
            target = instance.client.snapshot().game_frame + count
            sequence = instance.client.step(count)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                snap = instance.client.snapshot()
                if snap.ack_seq == sequence and snap.game_frame == target and snap.run_state_name == 'PAUSED':
                    break
                time.sleep(.001)
            else:
                raise TimeoutError(f'capture did not reach frame {target}')
            raw = instance.client.drain_frames()
            if len(raw) != count or snap.dropped_frames:
                raise RuntimeError('capture frame loss')
            native = events.drain()
            samples = {(e['frame'], e['attacker_seat']): e['native_inputs_before']
                       for e in native if e['kind'] == 'NATIVE_INPUT_SAMPLE'}
            for state in raw:
                if state.frameId != len(states):
                    raise RuntimeError('non-contiguous native frames')
                pair = [samples[(int(state.frameId), seat)] for seat in (0, 1)]
                inputs.append(dict(frame=int(state.frameId), native=pair,
                                   logical=[logical(v) for v in pair]))
                states.append(copy_state(state))
            contacts.extend(e for e in native if e['kind'] not in
                            ('NATIVE_INPUT_SAMPLE', 'NATIVE_INPUT_SCOPE'))
        with gzip.open(folder / 'native_frames.raw.gz', 'wb') as stream:
            for state in states:
                stream.write(ctypes.string_at(ctypes.addressof(state), ctypes.sizeof(state)))
        write(folder / 'frame_inputs.json', inputs)
        write(folder / 'native_events.json', contacts)
    finally:
        events.close()
        instance.close()
    return states, inputs, contacts


def candidates(contacts, inputs, states):
    roots = {(e['attacker_seat'], e['attack_instance']): e['action_after']
             for e in contacts if e['kind'] == 'ACTION_TRANSITION' and e['accepted']}
    groups = []
    active = {}
    for event in contacts:
        if event['kind'] != 'HIT_RESOLUTION' or event['hp_after'] >= event['hp_before']:
            continue
        key = (event['attacker_seat'], event['defender_seat'])
        # The resolver can reset a stale previous combo count during this hit.
        # A 6->1 result starts a new combo even when the entry count was nonzero.
        if (event['combo_count_before'] == 0 or key not in active
                or event['combo_count_after'] <= event['combo_count_before']):
            active[key] = []
            groups.append(active[key])
        active[key].append(event)
    selected = []
    for group in groups:
        if len(group) < 2:
            continue
        first, last = group[0]['frame'], group[-1]['frame']
        # Include the source action start, not only its first contact.
        source_roots = {(e['attacker_seat'], e['attack_instance']) for e in group}
        starts = [e['frame'] for e in contacts if e['kind'] == 'ACTION_TRANSITION'
                  and e['accepted'] and (e['attacker_seat'], e['attack_instance']) in source_roots]
        start = min(starts + [first])
        actions = [roots.get((e['attacker_seat'], e['attack_instance'])) for e in group]
        if any(a is None or not 300 <= a < 600 for a in actions):
            continue
        if any(row['logical'][seat][7] for row in inputs[start-1:last] for seat in (0, 1)):
            continue
        if any(600 <= getattr(state, player).actionId < 800 for state in states[start:last+1]
               for player in ('p1', 'p2')):
            continue
        selected.append(dict(start_frame=start, first_hit=first, last_hit=last,
            attacker=group[0]['attacker_seat'], hits=len(group), player_actions=actions,
            source_actions=[e['source_action'] for e in group],
            weather=sorted({int(state.activeWeather) for state in states[start:last+1]}),
            evidence='NATIVE_COMBO_COUNT_AND_CONTACT_CALLS', events=group))
    return selected


def rebuild_commands(module, decoded, neutral=False):
    """Use poll ticks offline; never assert that one poll equals one battle frame."""
    from copy import deepcopy
    replay = deepcopy(decoded)
    programs = []
    sides = [seat for seat in (0, 1) if replay['input_player_mask'] & (1 << seat)]
    for round_data in replay['rounds']:
        original = list(round_data['input_words'])
        schedules = {}
        for position, seat in enumerate(sides):
            values = [tuple(module.decode_keys(word).values()) for word in original[position::len(sides)]]
            if neutral:
                values = [(0,) * 8] * len(values)
            stages = compile_input_frames(values)
            player = CommandPlayer(stages)
            output = [player.next(tick, 1) for tick in range(len(values))]
            if output != values:
                raise RuntimeError('command interpreter changed poll-level logical keys')
            schedules[seat] = output
            programs.append(dict(round_index=round_data['round_index'], seat=seat,
                tick_semantics='NATIVE_INPUT_POLL_NOT_BATTLE_FRAME', samples=len(values),
                stages=[dict(direction=s.direction, buttons=s.buttons, frames=s.frames, command=s.command) for s in stages]))
        rebuilt = []
        for index, original_word in enumerate(original):
            seat = sides[index % len(sides)]
            values = schedules[seat][index // len(sides)]
            horizontal, vertical = values[:2]
            word = (4 if horizontal < 0 else 8 if horizontal > 0 else 0)
            word |= (1 if vertical < 0 else 2 if vertical > 0 else 0)
            word |= sum(value << bit for bit, value in enumerate(values[2:], 4))
            # Keep non-key bits and simultaneous-opposite raw directions only
            # for lossless source reconstruction; no invented key semantics.
            word |= original_word & ~0x3FF
            if not neutral:
                if original_word & 0xC == 0xC: word |= 0xC
                if original_word & 0x3 == 0x3: word |= 0x3
            rebuilt.append(word)
        round_data['input_words'] = rebuilt
    return module.encode(replay), programs


def reader_test(replay, expected, stop):
    os.environ['SOKURL_NATIVE_INPUT_COUNTERS'] = '0'
    instance = launch_replay_checkpoint(replay)
    events = CombatCaptureClient(instance.pid)
    result = dict(status='RUNNING', frames=0, first_difference=None,
                  scope='INPUTS_GENERATED_BY_COMMAND_PLAYER_NATIVE_REPLAY_READER_NOT_LIVE_VS')
    try:
        initial = battle_diff(expected[0], instance.state)
        if initial:
            return dict(status='BLOCKED', first_difference=dict(frame=0, fields=initial[:12]))
        events.drain()
        while result['frames'] < stop:
            count = min(64, stop-result['frames'])
            target = result['frames'] + count
            sequence = instance.client.step(count)
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                snap = instance.client.snapshot()
                if snap.ack_seq == sequence and snap.game_frame == target and snap.run_state_name == 'PAUSED':break
                time.sleep(.001)
            else:raise TimeoutError('generated-input playback stalled')
            raw = instance.client.drain_frames()
            native = events.drain()
            if len(raw) != count or snap.dropped_frames:raise RuntimeError('generated-input frame loss')
            for state in raw:
                result['frames'] += 1
                differences = state_diff(expected[result['frames']], state)
                if differences:
                    result.update(status='FAIL', first_difference=dict(frame=result['frames'],fields=differences[:12]))
                    return result
            result.setdefault('events', []).extend(e for e in native
                if e['kind'] not in ('NATIVE_INPUT_SAMPLE', 'NATIVE_INPUT_SCOPE'))
        result['status'] = 'PASS'
    finally:
        events.close()
        instance.close()
    return result


def noncard_diff(expected, actual):
    card_fields = ('cardGauge', 'cardCount', 'handIds', 'handCosts', 'selectedCardIndex')
    return [d for d in battle_diff(expected, actual, live_submode=True)
            if d['field'] != 'displayedWeather'
            if not any(d['field'].startswith(f'{seat}.{field}')
                       for seat in ('p1', 'p2') for field in card_fields)]


def rng_regions(pid, restore=None):
    """Known MT state from existing tools/validate_fresh_reset.py, frame0 only."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    signature = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                 ctypes.POINTER(ctypes.c_size_t)]
    kernel.ReadProcessMemory.argtypes = signature
    kernel.WriteProcessMemory.argtypes = signature
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x38 if restore else 0x10, False, pid)
    if not handle:raise OSError(ctypes.get_last_error(), 'initial RNG access failed')
    output = []
    try:
        for index, (address, size) in enumerate(((0x89B660, 624*4), (0x896B20, 4))):
            count = ctypes.c_size_t()
            buffer = ctypes.create_string_buffer(size)
            if restore:
                region = restore[index]
                data = bytes.fromhex(region['hex'])
                if region['address'] != address or len(data) != size or hashlib.sha256(data).hexdigest() != region['sha256']:
                    raise RuntimeError('RNG snapshot schema/hash mismatch')
                original = ctypes.create_string_buffer(data)
                if not kernel.WriteProcessMemory(handle, address, original, size, ctypes.byref(count)) or count.value != size:
                    raise OSError(ctypes.get_last_error(), 'initial RNG restore failed')
            if not kernel.ReadProcessMemory(handle, address, buffer, size, ctypes.byref(count)) or count.value != size:
                raise OSError(ctypes.get_last_error(), 'initial RNG read failed')
            data = buffer.raw
            output.append(dict(address=address, size=size, hex=data.hex(), sha256=hashlib.sha256(data).hexdigest()))
        if restore and output != restore:raise RuntimeError('RNG restore readback differs')
    finally:
        kernel.CloseHandle(handle)
    return output


def live_vs_test(decoded, states, inputs, stop, mode, initial_rng=None):
    if mode == 'native_counter_fallback':
        return dict(status='BLOCKED', mode=mode,
                    reason='Bridge StepWithInputs accepts only logical axes/buttons; raw duration counters are rejected')
    if any(600 <= getattr(state, seat).actionId < 800 for state in states[:stop+1] for seat in ('p1', 'p2')):
        return dict(status='BLOCKED', reason='prefix includes spell/system actions; retain original replay observation')
    round_data = decoded['rounds'][0]
    os.environ['SOKURL_NATIVE_INPUT_COUNTERS'] = '1' if mode == 'command_player' else '0'
    process = sokurl._launch_vs_from_title(35, headless=True, unlimited=True,
        seed=round_data['seed'], pause_at_start=True, characters=tuple(p['character'] for p in round_data['players']),
        cards_disabled=True, stage=int(states[0].stageId), music=round_data['music'])
    instance = events = client = None
    result = dict(status='RUNNING', mode=mode, frames=0,
        input_source='INDEPENDENT_LIVE_VS_NO_REPLAY_READER', first_difference=None,
        projection='ALL_RAW_BATTLE_FIELDS_EXCEPT_TRANSPORT_HASH_SEGMENT_SUBMODE_CARD_FIELDS_AND_FORECAST',
        hidden_state_equivalence='NOT_CLAIMED', forecast_differences=[],
        stage_source='NATIVE_RAW_FRAME_STAGE_ID_NOT_LEGACY_PARSER_LABEL',
        native_stage=int(states[0].stageId), parser_stage_label=round_data['stage'],
        removed_card_input_samples=sum(bool(row['logical'][seat][6] or row['logical'][seat][7])
                                      for row in inputs[:stop] for seat in (0, 1)))
    if initial_rng:
        result['projection'] = 'ALL_RAW_BATTLE_FIELDS_EXCEPT_TRANSPORT_HASH_SEGMENT_SUBMODE_AND_CARD_FIELDS'
    try:
        deadline = time.monotonic()+35
        while time.monotonic() < deadline:
            try:
                if client is None:client = BridgeClient(process.pid)
                snap = client.snapshot()
                if snap.in_gameplay and snap.game_frame == 0 and snap.run_state_name == 'PAUSED':break
            except BridgeUnavailable:
                pass
            time.sleep(.01)
        else:raise TimeoutError('live VS did not pause at frame0')
        instance = PracticeInstance(process, client, 0)
        events = CombatCaptureClient(process.pid)
        differences = noncard_diff(states[0], instance.state)
        if differences:
            result.update(status='BLOCKED', first_difference=dict(frame=0, fields=differences[:20]))
            return result
        if initial_rng:
            before_rng = rng_regions(process.pid)
            rng_regions(process.pid, initial_rng)
            result['initial_rng_sync'] = dict(at_frame=0, copied_bytes=2500,
                readback='PASS', original_vs_hashes=[r['sha256'] for r in before_rng],
                source_hashes=[r['sha256'] for r in initial_rng],
                boundary='INITIAL_STATE_ONLY_NO_FUTURE_STATE_COPY')
        events.drain()
        client.drain_frames()
        players = [CommandPlayer(compile_input_frames(tuple(row['logical'][seat][:6]) + (0, 0) for row in inputs[:stop]))
                   for seat in (0, 1)]
        if mode == 'command_player' and not events.native_input_counters:
            raise RuntimeError('counter capability missing in live VS')
        observed = []
        for index, row in enumerate(inputs[:stop]):
            pair = [p.next(index, 1, segment=0) for p in players] if mode == 'command_player' else (
                [tuple(values[:6])+(0, 0) for values in row['native']] if mode == 'native_counter_fallback' else [(0,) * 8] * 2)
            actual = instance.step(InputPair(tuple(pair[0]), tuple(pair[1])))
            native = events.drain()
            observed.extend(e for e in native if e['kind'] not in ('NATIVE_INPUT_SAMPLE', 'NATIVE_INPUT_SCOPE'))
            samples = {e['attacker_seat']: e['native_inputs_before'] for e in native if e['kind'] == 'NATIVE_INPUT_SAMPLE'}
            differences = noncard_diff(states[index+1], actual)
            if initial_rng and states[index+1].displayedWeather != actual.displayedWeather:
                differences.append(dict(field='displayedWeather', expected=int(states[index+1].displayedWeather),
                                        actual=int(actual.displayedWeather)))
            if states[index+1].displayedWeather != actual.displayedWeather and not result['forecast_differences']:
                result['forecast_differences'].append(dict(frame=index+1,
                    expected=int(states[index+1].displayedWeather), actual=int(actual.displayedWeather),
                    active_expected=int(states[index+1].activeWeather), active_actual=int(actual.activeWeather)))
            input_diff = {str(seat): dict(expected=row['native'][seat], actual=samples.get(seat))
                          for seat in (0, 1) if tuple(samples.get(seat, ())[:6]) != tuple(row['native'][seat][:6])}
            result['frames'] = index+1
            if differences or (input_diff and mode != 'neutral_negative_control'):
                result.update(status='FAIL', first_difference=dict(frame=index+1,
                    fields=differences[:20], native_inputs=input_diff))
                break
        else:result['status'] = 'PASS'
        result['events'] = observed
    except Exception as error:
        result.update(status='FAIL', error=repr(error))
    finally:
        if events:events.close()
        if instance:instance.close()
        else:
            if client:client.close()
            if process.is_running():sokurl.shutdown(5, process.pid)
    return result


def test(replay, states, inputs, stop, mode):
    if mode == 'native_counter_fallback':
        return dict(status='BLOCKED', mode=mode,
                    reason='StepWithInputs accepts only logical input; raw native duration values are not a valid command')
    os.environ['SOKURL_NATIVE_INPUT_COUNTERS'] = '1' if mode == 'command_player' else '0'
    instance = launch_replay_checkpoint(replay)
    events = CombatCaptureClient(instance.pid)
    result = dict(status='RUNNING', mode=mode, frames=0, first_difference=None,
                  projection='ALL_RAW_FIELDS_EXCEPT_HASH_SEGMENT_INPUT_TRANSPORT_AND_DECLARED_LIVE_SUBMODE',
                  input_source='BRIDGE_LIVE_OVERRIDE_WITH_REPLAY_READER_DISABLED')
    handle = None
    try:
        diff = battle_diff(states[0], instance.state)
        if diff:
            result.update(status='BLOCKED', first_difference=dict(frame=0, fields=diff[:12]))
            return result
        # SokuLib BattleMode.cpp / SokuAddresses.hpp: subMode is a uint8 at
        # 0x898688, offline VS is 3.1 and native replay is 3.2. Change only this
        # owned paused process's input-source mode; never patch EXE/DAT/state.
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WriteProcessMemory.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        kernel.ReadProcessMemory.argtypes = kernel.WriteProcessMemory.argtypes
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x38, False, instance.pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), 'owned process input-mode access failed')
        count = ctypes.c_size_t()
        current = ctypes.c_ubyte()
        if not kernel.ReadProcessMemory(handle, 0x898688, ctypes.byref(current), 1, ctypes.byref(count)) or current.value != 2:
            raise RuntimeError('expected native replay submode2')
        value = ctypes.c_ubyte(1)
        if not kernel.WriteProcessMemory(handle, 0x898688, ctypes.byref(value), 1, ctypes.byref(count)) or count.value != 1:
            raise OSError(ctypes.get_last_error(), 'live input-mode write failed')
        if not kernel.ReadProcessMemory(handle, 0x898688, ctypes.byref(current), 1, ctypes.byref(count)) or current.value != 1:
            raise RuntimeError('live input-mode readback failed')
        result['declared_mode_change'] = dict(address='0x898688', original=2, test=1)
        players = [CommandPlayer(compile_input_frames(row['logical'][seat] for row in inputs[:stop]))
                   for seat in (0, 1)]
        if mode == 'command_player' and not events.native_input_counters:
            raise RuntimeError('native input counter capability missing')
        events.drain()
        for index, row in enumerate(inputs[:stop]):
            if mode == 'command_player':
                pair = [p.next(index, 1, segment=0) for p in players]
            elif mode == 'native_counter_fallback':
                pair = row['native']
            else:
                pair = [(0,) * 8] * 2
            actual = instance.step(InputPair(tuple(pair[0]), tuple(pair[1])))
            native = events.drain()
            samples = {e['attacker_seat']: e['native_inputs_before'] for e in native
                       if e['kind'] == 'NATIVE_INPUT_SAMPLE'}
            if actual.battleSubMode != 1:
                raise RuntimeError('live input-mode reverted; replay input source is not isolated')
            diff = battle_diff(states[index+1], actual, live_submode=True)
            input_diff = {str(seat): dict(expected=row['native'][seat], actual=samples.get(seat))
                          for seat in (0, 1) if tuple(samples.get(seat, ())) != tuple(row['native'][seat])}
            result['frames'] = index + 1
            if diff or (input_diff and mode != 'neutral_negative_control'):
                result.update(status='FAIL', first_difference=dict(frame=index+1,
                    fields=diff[:12], native_inputs=input_diff))
                break
        else:
            result['status'] = 'PASS'
    finally:
        if handle:
            value = ctypes.c_ubyte(2)
            kernel.WriteProcessMemory(handle, 0x898688, ctypes.byref(value), 1, ctypes.byref(count))
            kernel.CloseHandle(handle)
        events.close()
        instance.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replays', nargs='+', type=Path)
    parser.add_argument('--parser', type=Path,
                        help='existing export_native_replay_inputs.py, reused read-only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=2500)
    parser.add_argument('--reuse-capture', type=Path,
                        help='existing native prefix evidence, verified by source SHA256')
    parser.add_argument('--compiled-reader', action='store_true',
                        help='verify input-equivalent command output with original replay reader')
    parser.add_argument('--live-vs', action='store_true', help='independent cold VS, cards disabled; no state patches')
    parser.add_argument('--sync-initial-rng', action='store_true',
                        help='explicitly align only known original frame0 RNG regions before live VS')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.parser:
        spec = importlib.util.spec_from_file_location('existing_replay_parser', args.parser)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from sokurl_platform import replay_inputs as module
    os.environ.update(SOKURL_COMBAT_EVENTS='1', SOKURL_CARDS_DISABLED='0',
                      SOKURL_HEADLESS_RENDER='1', SOKURL_UNLIMITED_PACING='1', SOKURL_MUTE_AUDIO='1')
    for name in ('SOKURL_VS_BOOTSTRAP', 'SOKURL_FIXED_WEATHER', 'SOKURL_FRESH_RESET', 'SOKURL_CAPTURE_IMAGES'):
        os.environ.pop(name, None)
    summary = []
    try:
        for number, replay in enumerate(args.replays):
            folder = args.output / f'replay_{number+1}'
            folder.mkdir()
            data = replay.read_bytes()
            decoded = module.decode(data)
            if module.encode(decoded) != data:
                raise RuntimeError('existing parser byte roundtrip failed')
            write(folder / 'replay_inputs.json', dict(source=str(replay.resolve()),
                  sha256=hashlib.sha256(data).hexdigest(), **decoded,
                  index_semantics='NATIVE_INPUT_POLL_NOT_SIMULATION_FRAME',
                  conditions='ORIGINAL_REPLAY_WEATHER_AND_CARDS_UNMODIFIED'))
            if args.reuse_capture:
                existing = args.reuse_capture / f'replay_{number+1}'
                metadata = json.loads((existing / 'replay_inputs.json').read_text(encoding='utf-8'))
                if metadata['sha256'] != hashlib.sha256(data).hexdigest():
                    raise RuntimeError('capture/replay SHA256 mismatch')
                payload = gzip.decompress((existing / 'native_frames.raw.gz').read_bytes())
                size = ctypes.sizeof(RawFrameState)
                if len(payload) % size:
                    raise RuntimeError('truncated native state capture')
                states = [RawFrameState.from_buffer_copy(payload[offset:offset+size])
                          for offset in range(0, len(payload), size)]
                inputs = json.loads((existing / 'frame_inputs.json').read_text(encoding='utf-8'))
                contacts = json.loads((existing / 'native_events.json').read_text(encoding='utf-8'))
                if len(states) != len(inputs)+1:
                    raise RuntimeError('native frame/input count mismatch')
                write(folder / 'capture_reference.json', dict(path=str(existing.resolve()),
                    frames=len(inputs), replay_sha256=metadata['sha256'],
                    raw_sha256=hashlib.sha256(payload).hexdigest()))
            else:
                states, inputs, contacts = capture(replay, args.frames, folder)
            cases = candidates(contacts, inputs, states)
            write(folder / 'nonspell_combo_candidates.json', cases)
            item = dict(replay=str(replay), candidates=len(cases), tests=[])
            # Test the first eligible combo from the replay's true initial state.
            if cases:
                chosen = next((case for case in cases if len(set(case['player_actions'])) > 1), cases[0])
                item['reference_case'] = chosen
                stop = min(args.frames, chosen['last_hit'] + 60)
                if args.live_vs:
                    programs = []
                    for seat in (0, 1):
                        stages = compile_input_frames(tuple(row['logical'][seat][:6])+(0, 0) for row in inputs[:stop])
                        programs.append(dict(seat=seat, tick_semantics='SIMULATION_FRAME',
                            facing_argument=1, total_frames=stop,
                            stages=[dict(direction=s.direction, buttons=s.buttons, frames=s.frames, command=s.command) for s in stages]))
                    write(folder / 'live_command_programs.json', programs)
                    initial_rng = None
                    if args.sync_initial_rng:
                        os.environ['SOKURL_NATIVE_INPUT_COUNTERS'] = '0'
                        origin = launch_replay_checkpoint(replay)
                        try:
                            if state_diff(states[0], origin.state):
                                raise RuntimeError('new original frame0 differs from captured reference')
                            initial_rng = rng_regions(origin.pid)
                            write(folder / 'initial_rng.json', dict(frame=0, regions=initial_rng,
                                replay_sha256=hashlib.sha256(data).hexdigest(),
                                evidence='KNOWN_NATIVE_MT_STATE_LAYOUT_FROM_EXISTING_RESET_VALIDATOR'))
                        finally:
                            origin.close()
                    for mode in ('command_player', 'native_counter_fallback', 'neutral_negative_control'):
                        outcome = live_vs_test(decoded, states, inputs, stop, mode, initial_rng)
                        write(folder / f'live_vs_{mode}.json', outcome)
                        item['tests'].append({k: v for k, v in outcome.items() if k != 'events'})
                    positive = item['tests'][0]['status'] == 'PASS'
                    negative = item['tests'][2]['status'] == 'FAIL'
                    item['status'] = 'PASS_INDEPENDENT_NONCARD_COMBAT' if positive and negative else 'FAIL'
                elif args.compiled_reader:
                    output, programs = rebuild_commands(module, decoded)
                    if output != data:raise RuntimeError('command-generated replay is not lossless')
                    compiled = folder / 'command_generated.rep'
                    compiled.write_bytes(output)
                    write(folder / 'command_programs.json', programs)
                    outcome = reader_test(compiled, states, len(inputs))
                    write(folder / 'compiled_reader.json', outcome)
                    item['tests'].append({k: v for k, v in outcome.items() if k != 'events'})
                    neutral, _ = rebuild_commands(module, decoded, neutral=True)
                    changed = folder / 'neutral_negative.rep'
                    changed.write_bytes(neutral)
                    negative = reader_test(changed, states, stop)
                    write(folder / 'neutral_reader.json', negative)
                    item['tests'].append({k: v for k, v in negative.items() if k != 'events'})
                    item['byte_identical_generated_inputs'] = True
                    item['negative_control_rejects_original_trajectory'] = negative['status'] == 'FAIL'
                    item['status'] = 'PASS_INPUT_EQUIVALENCE' if outcome['status'] == 'PASS' and negative['status'] == 'FAIL' else 'FAIL'
                    item['independent_live_input_injection'] = 'BLOCKED_BY_REPLAY_INPUT_MANAGER'
                else:
                    for mode in ('command_player', 'native_counter_fallback', 'neutral_negative_control'):
                        outcome = test(replay, states, inputs, stop, mode)
                        write(folder / f'{mode}.json', outcome)
                        item['tests'].append(outcome)
            else:
                item['status'] = 'BLOCKED_NO_ELIGIBLE_NON_SPELL_COMBO_IN_PREFIX'
            summary.append(item)
            write(args.output / 'summary.json', summary)
            print(json.dumps({k: v for k, v in item.items() if k != 'reference_case'}), flush=True)
    finally:
        write(args.output / 'summary.json', summary)
    if (args.live_vs or args.compiled_reader) and any(not item.get('status', '').startswith('PASS') for item in summary):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

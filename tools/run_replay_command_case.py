"""Run a validated command fixture in live VS, without loading any .rep file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from bridge_shared import BridgeClient, BridgeUnavailable
from combat_capture import CombatCaptureClient
from command_input import CommandPlayer, Keys
from frame_validation import PracticeInstance, InputPair
from probe_replay_commands import rng_regions
from build_replay_command_cases import CONTACT_FIELDS, combat_projection
import sokurl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--neutral-negative', action='store_true')
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior acceptance evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = args.case.read_bytes()
    case = json.loads(data)
    if case['status'] != 'PASS_COMBAT_PROJECTION_AND_ORDERED_CONTACTS' or case['start_frame'] != 0:
        raise ValueError('validated frame0 fixture required')
    if type(case['stop_frame']) is not int or not 1 <= case['stop_frame'] <= 10000:
        raise ValueError('bounded fixture frame budget required')
    programs = [CommandPlayer(tuple(Keys(**stage) for stage in program['stages'])) for program in case['programs']]
    if len(programs) != 2 or [p['seat'] for p in case['programs']] != [0, 1]:
        raise ValueError('ordered two-seat fixture required')
    os.environ.update(SOKURL_COMBAT_EVENTS='1', SOKURL_NATIVE_INPUT_COUNTERS='1',
        SOKURL_CARDS_DISABLED='1', SOKURL_MUTE_AUDIO='1')
    for key in ('SOKURL_FIXED_WEATHER', 'SOKURL_FRESH_RESET', 'SOKURL_CAPTURE_IMAGES'):
        os.environ.pop(key, None)
    process = sokurl._launch_vs_from_title(35, headless=True, unlimited=True,
        seed=case['seed'], pause_at_start=True, characters=tuple(case['characters']),
        cards_disabled=True, stage=case['native_stage'])
    client = instance = events = None
    result = dict(status='RUNNING', fixture_sha256=hashlib.sha256(data).hexdigest(),
        source_replay_sha256=case['source_sha256'], frames=0,
        input_source='LIVE_VS_SHARED_COMMAND_PLAYER_NO_REPLAY_FILE_LOADED',
        neutral_negative=args.neutral_negative, original_next_state_copy=False)
    start = time.monotonic()
    try:
        deadline = time.monotonic()+35
        while time.monotonic() < deadline:
            try:
                if client is None:client=BridgeClient(process.pid)
                snapshot = client.snapshot()
                if snapshot.in_gameplay and snapshot.game_frame == 0 and snapshot.run_state_name == 'PAUSED':break
            except BridgeUnavailable:pass
            time.sleep(.01)
        else:raise TimeoutError('live VS frame0 unavailable')
        instance = PracticeInstance(process, client, 0)
        events = CombatCaptureClient(process.pid)
        if not events.native_input_counters:raise RuntimeError('enabled ABI8 input counters required')
        rng_regions(process.pid, case['initial_rng']['regions'])
        result['initial_rng_readback'] = 'PASS'
        events.drain()
        client.drain_frames()
        observed = []
        final_inputs = None
        for index in range(case['stop_frame']):
            pair = [(0,) * 8] * 2 if args.neutral_negative else [p.next(index, 1) for p in programs]
            state = instance.step(InputPair(tuple(pair[0]), tuple(pair[1])))
            native = events.drain()
            sample = {e['attacker_seat']: e['native_inputs_before'] for e in native if e['kind'] == 'NATIVE_INPUT_SAMPLE'}
            final_inputs = [list(sample[seat]) for seat in (0, 1)]
            candidate = case['combo']
            observed.extend({name:e.get(name) for name in CONTACT_FIELDS} for e in native
                if e['kind'] == 'HIT_RESOLUTION' and candidate['first_hit'] <= e['frame'] <= candidate['last_hit']
                and e['attacker_seat'] == candidate['attacker'] and e['hp_after'] < e['hp_before'])
            result['frames'] = index+1
        actual = combat_projection(state)
        differences = [dict(field=key, expected=value, actual=actual.get(key))
                       for key, value in case['final_combat_projection'].items() if actual.get(key) != value]
        result.update(ordered_contacts_match=observed == case['ordered_contacts'],
            contacts=observed, final_inputs_match=final_inputs == case['final_native_inputs'],
            final_projection_differences=differences[:20])
        if differences or not result['ordered_contacts_match'] or not result['final_inputs_match']:
            raise RuntimeError('fixture contact/input/final combat projection differs')
        if not all(p.done for p in programs):raise RuntimeError('fixture programs not fully consumed')
        result['status'] = 'PASS'
    except Exception as error:
        result.update(status='FAIL', error=repr(error))
    finally:
        if events:events.close()
        if instance:instance.close()
        else:
            if client:client.close()
            if process.is_running():sokurl.shutdown(5, process.pid)
        result['seconds'] = time.monotonic()-start
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'contacts'}, indent=2))
    if result['status'] != 'PASS':raise SystemExit(1)


if __name__ == '__main__':main()

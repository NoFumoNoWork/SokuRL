"""Package already validated non-spell replay cases; does not run the game."""
import argparse
import hashlib
import json
import ctypes
import gzip
from pathlib import Path

from sokurl_platform.command_input import compile_input_frames
from sokurl_platform.bridge_shared import RawFrameState
from sokurl_platform.frame_validation import flatten_ctypes


CONTACT_FIELDS = ('frame', 'attacker_seat', 'defender_seat', 'source_action',
    'source_sequence', 'hp_before', 'hp_after', 'spirit_before', 'spirit_after',
    'rate_before', 'rate_at_damage', 'rate_after', 'combo_count_before',
    'combo_count_after', 'limit_before', 'limit_after', 'defender_action_before',
    'defender_action_after', 'counter_hit_first')


def combat_projection(state):
    excluded = {'stateHash', 'segmentId', 'battleSubMode'}
    prefixes = tuple(f'{seat}.{field}' for seat in ('p1', 'p2') for field in
                     ('input.', 'cardGauge', 'cardCount', 'handIds', 'handCosts', 'selectedCardIndex'))
    return {key: value for key, value in flatten_ctypes(state).items()
            if key not in excluded and not key.startswith(prefixes)}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, summary in enumerate(load(args.validation / 'summary.json'), 1):
        source = args.capture / f'replay_{index}'
        validated = args.validation / f'replay_{index}'
        result = load(validated / 'live_vs_command_player.json')
        if result['status'] != 'PASS':raise RuntimeError('only actual live-VS PASS may be packaged')
        candidate = summary['reference_case']
        native = load(source / 'native_events.json')
        selected = lambda events: [{name:e.get(name) for name in CONTACT_FIELDS} for e in events
            if e['kind'] == 'HIT_RESOLUTION' and candidate['first_hit'] <= e['frame'] <= candidate['last_hit']
            and e['attacker_seat'] == candidate['attacker'] and e['hp_after'] < e['hp_before']]
        original, actual = selected(native), selected(result['events'])
        if original != actual:raise RuntimeError('ordered contact outcome parity failed')
        stop = result['frames']
        inputs = load(source / 'frame_inputs.json')[:stop]
        raw = gzip.decompress((source / 'native_frames.raw.gz').read_bytes())
        size = ctypes.sizeof(RawFrameState)
        final = RawFrameState.from_buffer_copy(raw[stop*size:(stop+1)*size])
        programs = []
        for seat in (0, 1):
            stages = compile_input_frames(tuple(row['logical'][seat][:6])+(0, 0) for row in inputs)
            programs.append(dict(seat=seat, facing_argument=1,
                stages=[dict(direction=s.direction, buttons=s.buttons, frames=s.frames, command=s.command) for s in stages]))
        metadata = load(source / 'replay_inputs.json')
        rng = load(validated / 'initial_rng.json')
        case = dict(status='PASS_COMBAT_PROJECTION_AND_ORDERED_CONTACTS',
            source_replay=metadata['source'], source_sha256=metadata['sha256'],
            start_frame=0, stop_frame=stop, combo=candidate,
            tick_semantics='SIMULATION_FRAME', absolute_directions=True, cards_disabled=True,
            native_stage=result['native_stage'], seed=metadata['rounds'][0]['seed'],
            characters=[p['character'] for p in metadata['rounds'][0]['players']],
            input_transport='BRIDGE_STEP_WITH_INPUTS_NATIVE_COUNTERS_ENABLED',
            initial_rng=rng, programs=programs, ordered_contacts=original,
            final_combat_projection=combat_projection(final),
            final_native_inputs=[tuple(values[:6])+(0, 0) for values in inputs[-1]['native']],
            validation_report=str((validated / 'live_vs_command_player.json').resolve()),
            validation_report_sha256=hashlib.sha256((validated / 'live_vs_command_player.json').read_bytes()).hexdigest(),
            boundary='FRAME0_RNG_INITIALIZATION_ONLY; NO_REPLAY_READER_OR_FUTURE_STATE_COPY; NOT_FULL_HIDDEN_STATE_EQUIVALENCE')
        folder = args.output / f'replay_{index}'
        folder.mkdir()
        (folder / 'case.json').write_text(json.dumps(case, indent=2), encoding='utf-8')
        rows.append(dict(replay=index, status=case['status'], frames=stop,
                         combo_start=candidate['start_frame'], combo_end=candidate['last_hit'],
                         actions=candidate['player_actions'], ordered_contact_count=len(original),
                         fixture=str((folder / 'case.json').resolve())))
    (args.output / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':main()

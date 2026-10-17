"""Observe untouched native replays with existing Platform APIs; no BC labels.

This bounded preflight deliberately fails closed when poll-to-frame identity is
not available. State/input observations are diagnostic evidence, not supervision.
"""
import argparse
from contextlib import ExitStack
import ctypes
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from sokurl_platform.combat_capture import CombatCaptureClient
from sokurl_platform.images import ImageClient
from sokurl_platform.replay import launch_replay
from sokurl_platform.replay_format import Replay
from sokurl_platform import sokurl
from sokurl_platform.native_status import remaining_replay_inputs
from sokurl_platform.memory import ProcessMemory
from soku_rl.replay.human import consumed_pair, verify_pair

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding='utf-8')


def player_state(player):
    return dict(character=int(player.characterId), action=int(player.actionId),
        sequence=int(player.sequenceId), hp=int(player.hp), spirit=int(player.spirit),
        hitstop=int(player.hitstop), skill_levels=list(player.skillLevels),
        card_count=int(player.cardCount), hand=list(player.handIds))


def inspect(args):
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'rl/runs') or output.exists():
        raise ValueError('use a NEW output under rl/runs')
    if not 1 <= args.frames <= 1024:
        raise ValueError('bounded pilot prefix required')
    plan = json.loads((args.audit / 'pilot.json').read_text())
    audit = json.loads((args.audit / 'audit.json').read_text())
    if audit['status'] != 'PASS' or len(plan['rows']) != 20:
        raise ValueError('requires audited20-replay training-only pilot')
    output.mkdir(parents=True)
    report = dict(status='RUNNING', purpose='ORIGINAL_OBSERVATION_PREFLIGHT_NOT_SUPERVISION',
        frame_limit=args.frames, replays=[], labels_generated=0,
        input_source='native REP reader; no step_with_inputs, weather override, card suppression or skill writes',
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), ROOT/'platform/src/sokurl_platform/replay.py',
            ROOT/'platform/src/sokurl_platform/combat_capture.py',
            ROOT/'platform/native/SokuRLBridge/CombatCapture.cpp')})
    save(output / 'report.json', report)
    before = {str(p): digest(p) for p in (sokurl.GAME_EXE, sokurl.SKIPINTRO_INI,
              sokurl.GAME_DIR/'config123.dat')}
    try:
        with patch.dict(os.environ, {'SOKURL_CARDS_DISABLED':'0','SOKURL_FRESH_RESET':'0',
                'SOKURL_NATIVE_INPUT_COUNTERS':'0','SOKURL_NATIVE_TRACE':'0',
                'SOKURL_ORIGINAL_REPLAY_RECORDING':'0','SOKURL_COMBAT_EVENTS':'1',
                'SOKURL_HEADLESS_RENDER':'1','SOKURL_MUTE_AUDIO':'1'}):
            for row in plan['rows'][:args.limit]:
                path = Path(row['file'])
                if digest(path) != row['sha256']:
                    raise ValueError('REP source hash changed')
                replay = Replay.decode(path.read_bytes())
                folder = output / str(row['id'])
                folder.mkdir()
                started = time.perf_counter()
                details = dict(id=row['id'], sha256=row['sha256'], source=str(path),
                    capture_status='RUNNING', supervision_status='BLOCKED',
                    reason='Candidate consumption intervals and postprocessed key samples have not passed decision-label acceptance',
                    rounds_in_rep=len(replay.matches), full_replay_parity='NOT_TESTED',
                    original_frames='UNKNOWN; REP words count input polls,not battle frames',
                    supervised_frames=0, eligible_segments=0, captured_frames=0,
                    frame_alignment='CANDIDATE_NATIVE_CONSUMPTION_INTERVALS_WITH_POSTPROCESSING_CHECK',
                    observation_mode='existing Platform renderer state; no pixels', examples=[])
                events_count, input_counts = {}, []
                alignment_errors, aligned_polls, no_poll_frames = [], 0, 0
                with ExitStack() as resources:
                    game = launch_replay(path, 45., unlimited=True, observation_mode='state')
                    resources.callback(game.close)
                    events = CombatCaptureClient(game.pid)
                    resources.callback(events.close)
                    image = ImageClient(game.pid)
                    resources.callback(image.close)
                    memory = ProcessMemory(game.pid)
                    resources.callback(memory.close)
                    details['combat_abi'] = events.version
                    details['replay_input_capture'] = events.replay_input_capture
                    events.drain()
                    raw = game.state
                    actual = (int(raw.p1.characterId), int(raw.p2.characterId))
                    expected = tuple(p.character for p in replay.matches[0].players)
                    if actual != expected:
                        raise ValueError('native/REP seats differ')
                    details['characters'] = list(actual)
                    words = replay.matches[0].input_words
                    mask = replay.header[8]
                    remaining_before = None
                    details['initial'] = dict(weather=int(raw.activeWeather),
                        players=[player_state(raw.p1),player_state(raw.p2)])
                    with gzip.open(folder/'frames.raw.gz','wb') as states, gzip.open(
                            folder/'observed.jsonl.gz','wt',encoding='utf-8') as observations:
                        states.write(bytes(raw))
                        for index in range(args.frames+1):
                            if index:
                                raw = game.step_native(5.)
                                states.write(bytes(raw))
                            native = events.drain()
                            for event in native:
                                kind = event['kind']
                                events_count[kind] = events_count.get(kind,0)+1
                            render = image.read_state(index, 3.)
                            if int(raw.frameId) != index:
                                raise ValueError('native frame discontinuity')
                            samples = [e for e in native if e['kind']=='NATIVE_INPUT_SAMPLE']
                            memory.begin_frame()
                            remaining_after = remaining_replay_inputs(memory)
                            pair = None
                            if index:
                                try:
                                    pair = consumed_pair(words, mask, remaining_before, remaining_after)
                                    if pair is None:
                                        no_poll_frames += 1
                                    else:
                                        verify_pair(pair, samples, index)
                                        aligned_polls += 1
                                except ValueError as error:
                                    alignment_errors.append(dict(frame=index, before=remaining_before,
                                        after=remaining_after, reason=str(error), pair=pair, samples=samples))
                            remaining_before = remaining_after
                            if index:
                                input_counts.append(len(samples))
                            record = dict(frame=index, segment=int(raw.segmentId),
                                weather=int(raw.activeWeather), players=[player_state(raw.p1),player_state(raw.p2)],
                                render=asdict(render), native_events=native,
                                remaining_replay_words=remaining_after,
                                consumed_pair=pair, supervision_mask=False)
                            observations.write(json.dumps(record,ensure_ascii=True)+'\n')
                            if (index <= 4 or any(p.hitstop for p in (raw.p1,raw.p2))
                                    or any(e['kind'].endswith('RESOLUTION') for e in native)) and len(details['examples'])<16:
                                details['examples'].append(record)
                            details['captured_frames'] = index
                            if game.client.snapshot().dropped_frames:
                                raise RuntimeError('native frame loss')
                    details.update(capture_status='PASS', event_counts=events_count,
                        aligned_polls=aligned_polls, no_poll_frames=no_poll_frames,
                        alignment_errors=alignment_errors,
                        count_alignment_status='FAIL' if alignment_errors else 'PASS',
                        input_samples_per_frame=dict((str(n),input_counts.count(n)) for n in set(input_counts)),
                        raw_sha256=digest(folder/'frames.raw.gz'), observed_sha256=digest(folder/'observed.jsonl.gz'),
                        elapsed_seconds=time.perf_counter()-started,
                        diagnostic_frames_per_second=args.frames/(time.perf_counter()-started))
                save(folder/'result.json', details)
                report['replays'].append(details)
                save(output/'report.json',report)
                print(json.dumps(dict(id=row['id'],capture=details['capture_status'],frames=details['captured_frames'],
                                      supervision=details['supervision_status'])),flush=True)
        report['status']='BLOCKED'
        report['blocker']='Raw-consumption/postprocessing and full replay/round/condition acceptance not closed. No guessed labels or training.'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        report['protected_files_unchanged']={p:digest(Path(p))==h for p,h in before.items()}
        save(output/'report.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=256)
    parser.add_argument('--limit',type=int,choices=range(1,21),default=20)
    result = inspect(parser.parse_args())
    raise SystemExit(2 if result['status']=='BLOCKED' else 0)

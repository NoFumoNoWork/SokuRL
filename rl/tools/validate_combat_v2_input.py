"""Bounded native 1F injection/reset proof; does not infer command recognition."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from bridge_shared import BridgeClient
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.vector_env import TwoPlayerVectorEnv
from soku_rl.env.encoding import AGENTS, encode_action
from validate_platform_integration import VISIBILITY


def program():
    neutral = (0, 0, 0, 0, 0, 0, 0, 0)
    # Keep startup frames distinct from command windows; raw frame zero need not
    # already be an actionable fighting state.
    commands = [neutral] * 600
    chunks = {
        'direction_c': [(1, 0, 0, 0, 1, 0, 0, 0), neutral],
        'quarter_circle_b': [(0, 1, 0, 0, 0, 0, 0, 0),
                             (1, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0, 0, 0), neutral],
        'double_down_b': [(0, 1, 0, 0, 0, 0, 0, 0), neutral,
                          (0, 1, 0, 1, 0, 0, 0, 0), neutral],
        'a_hold_release_repeat': [(0, 0, 1, 0, 0, 0, 0, 0)] * 3 +
                                [neutral, (0, 0, 1, 0, 0, 0, 0, 0), neutral,
                                 (0, 0, 1, 0, 0, 0, 0, 0), neutral],
    }
    windows = []
    for name, chunk in chunks.items():
        windows.append(dict(name=name, first_frame=len(commands)+1, submitted_frames=len(chunk)))
        commands.extend(chunk)
        commands.extend([neutral] * 48)
    return commands, windows


def run(output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ['SOKURL_RL_ROOT'])
    backend = WorkerBackend([sys.executable, '-B', str(root/'tools/rollout_worker.py')],
        os.environ['SOKURL_PLATFORM_ROOT'], str(output.with_suffix('.worker.log')), 120, 40,
        True, os.environ['SOKURL_GAME_DIR'], fresh_reset=True,
        cards_disabled=True, fixed_weather='clear')
    config = EpisodeConfig(7200, 8, 1, 0, 'animation', VisibilityConfig(**VISIBILITY), LEGACY_MATCH)
    backend.configure_observation(config.backend_observation())
    env = TwoPlayerVectorEnv(backend, 2, config)
    clients = {}
    report = dict(status='RUNNING', command_recognition='NOT_VERIFIED',
                  game_semantics='UNCHANGED', animation_permission='PRIVILEGED_PROXY')
    commands, report['windows'] = program()
    start = time.monotonic()
    def execute():
        records = []
        for frame, held in enumerate(commands, 1):
            actions = {slot: {AGENTS[0]: encode_action(held), AGENTS[1]: encode_action((0,)*8)}
                       for slot in range(2)}
            obs, _, terms, truncs, _ = env.step(actions)
            row = []
            for slot in range(2):
                raw = clients[slot].snapshot().latest
                actual = tuple(getattr(raw.p1.input, key) for key in
                    ('horizontalAxis','verticalAxis','a','b','c','d','changeCard','spellcard'))
                assert int(raw.frameId) == frame and actual == held
                assert not any(terms[slot].values()) and not any(truncs[slot].values())
                assert np.isfinite(obs[slot][AGENTS[0]]).all()
                row.append(dict(hash=f'{raw.stateHash:016X}', action=int(raw.p1.actionId),
                                sequence=int(raw.p1.sequenceId), consumed=actual,
                                players=[{key:getattr(p,key) for key,_ in p._fields_
                                          if isinstance(getattr(p,key),(int,float))}
                                     for p in (raw.p1, raw.p2)],
                                world={key:getattr(raw,key) for key,_ in raw._fields_
                                       if isinstance(getattr(raw,key),(int,float)) and key != 'stateHash'},
                                object_counts=[raw.p1ObjectCount,raw.p2ObjectCount]))
            records.append(row)
        return records
    try:
        env.reset({0:1732, 1:1733})
        for slot, status in backend.request('status', [0,1]).items():
            clients[slot] = BridgeClient(status['pid'])
        first = execute()
        report['first'] = first
        env.reset({0:1732, 1:1733})
        second = execute()
        report['second'] = second
        report['first_difference_frame'] = next((i+1 for i,(a,b) in enumerate(zip(first,second)) if a!=b), None)
        report['first_native_hash_difference_frame'] = next((i+1 for i,(a,b) in enumerate(zip(first,second))
            if any(x['hash'] != y['hash'] for x,y in zip(a,b))), None)
        # segmentId is the reset epoch, not a simulated-state parity field.
        assert all(all((x['hash'],x['consumed']) == (y['hash'],y['consumed']) for x,y in zip(a,b))
                   for a,b in zip(first,second)), 'same-seed actual input/native trajectory differs'
        report.update(status='PASS', frames_per_run=len(commands), instances=2,
                      checks=['one_frame_advancement','held_release_direction_input',
                              'same_seed_native_state_replay'], records=first)
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        for client in clients.values():
            client.close()
        env.close()
        report['seconds'] = time.monotonic()-start
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: 1F native consumed input and same-seed replay; recognition not asserted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)

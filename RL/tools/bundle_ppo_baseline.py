"""Join completed cases, preserving failed-run status and strict overlap checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--retry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a, b = load(args.original/'result.json'), load(args.retry/'result.json')
    if a['success'] or not b['success']:
        raise RuntimeError('expected retained failed original and completed retry')
    af, bf = [row['runtime']['fingerprints'] for row in (a, b)]
    if af['game_id'] != bf['game_id']:
        raise RuntimeError('game/module identity differs')
    permitted = {'RL/tools/run_ppo_control.py', 'RL/tools/summarize_ppo_control.py'}
    changed = [key for key in af['source_hashes'].keys() & bf['source_hashes'].keys()
               if af['source_hashes'][key] != bf['source_hashes'][key]]
    if set(changed) - permitted:
        raise RuntimeError(f'execution source differs: {changed}')
    old = load(args.original/'progress.json')['games']
    new = b['result']['games']
    merged = {g['trial_id']: (g, args.original) for g in old}
    if len(merged) != len(old):
        raise RuntimeError('duplicate original trial')
    overlap = 0
    for game in new:
        identifier = game['trial_id']
        if identifier in merged:
            previous, source = merged[identifier]
            # All recorded semantics, not merely outcome. Only file provenance differs.
            if {k:v for k,v in previous.items() if k != 'replay'} != {
                    k:v for k,v in game.items() if k != 'replay'}:
                raise RuntimeError(f'overlap record differs: {identifier}')
            with np.load(source/previous['replay']) as x, np.load(args.retry/game['replay']) as y:
                if not np.array_equal(x['actions'], y['actions']):
                    raise RuntimeError(f'overlap actions differ: {identifier}')
            overlap += 1
        else:
            merged[identifier] = (game, args.retry)
    plan = load(args.original/'plan.json')
    if set(merged) != {g['trial_id'] for g in plan} or len(merged) != len(plan):
        raise RuntimeError('original planned coverage not complete')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    games = [game | {'replay': os.path.relpath(source/game['replay'], output),
                    'case_source': str(source.resolve())} for game, source in merged.values()]
    report = dict(success=True, phase='case_coverage_complete',
        meaning='Completed-case coverage, NOT original eight-instance soak PASS',
        environment_reliability={'original_eight_instance': 'FAIL_RESET_TIMEOUT',
                                 'retry_four_instance': 'PASS_32_CASES_ONLY'},
        overlap_cases=overlap, overlap='Exact recorded fields and all action indices; not full hidden-state parity',
        allowed_non_execution_source_changes=changed,
        source_result_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (args.original/'result.json', args.retry/'result.json')},
        runtime=b['runtime'], result={'games': games})
    (output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'CASE COVERAGE PASS:{len(games)}; exact overlaps:{overlap}; eight-instance reliability remains FAIL')


if __name__ == '__main__':
    main()

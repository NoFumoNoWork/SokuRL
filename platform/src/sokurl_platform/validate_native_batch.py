"""Bounded original-engine reset/isolation parity without importing learning code."""
import argparse
import json
from pathlib import Path
import sys
import time

from .native_batch import NativeGameBatch, NativeInput
from .match import LEGACY_MATCH
from .startup import configure_game
from .resources import game_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--frames', type=int, default=160)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve evidence')
    if args.workers < 2 or args.frames < 1:
        raise ValueError('at least two workers and a positive frame budget required')
    configure_game(game_directory())
    backend = NativeGameBatch(40)
    backend.fresh_reset, backend.cards_disabled, backend.fixed_weather = True, True, 'clear'
    backend.configure(LEGACY_MATCH, max_frames=args.frames + 100)
    seeds = {i: 1732 + i * 1009 for i in range(args.workers)}
    report = dict(status='RUNNING', workers=args.workers, frames=args.frames)
    started = time.monotonic()

    def trajectory():
        result = []
        for frame in range(args.frames):
            left = NativeInput((1 if frame % 80 < 40 else -1, 0, int(frame % 30 < 3), 0, 0, 0, 0, 0))
            right = NativeInput((-1 if frame % 80 < 40 else 1, 0, 0, int(frame % 40 < 3), 0, 0, 0, 0))
            states = backend.step({i: (left, right) for i in seeds})
            result.append({i: int(state.raw.stateHash) for i, state in states.items()})
        return result

    try:
        backend.reset_slots(seeds)
        pids = {i: p.pid for i, p in backend.processes.items()}
        before = trajectory()
        backend.reset_slots(seeds)
        after = trajectory()
        if before != after:
            first = next(i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1])
            raise AssertionError(f'same-seed state-hash divergence at frame {first+1}')
        untouched = backend.status(list(seeds)[1:])
        new_seed = seeds[0] + 17
        state = backend.reset_slots({0: new_seed})[0]
        if state.frame != 0 or state.raw.randomSeed != new_seed:
            raise AssertionError('new seed was not applied at frame zero')
        if backend.status(list(seeds)[1:]) != untouched:
            raise AssertionError('partial reset changed other instances')
        if {i: p.pid for i, p in backend.processes.items()} != pids:
            raise AssertionError('fresh reset recreated a game process')
        forbidden = [n for n in sys.modules if n.split('.')[0] in ('soku_rl','torch','gymnasium','stable_baselines3')]
        if forbidden:
            raise AssertionError(f'Platform imported learning dependencies: {forbidden}')
        report.update(status='PASS', same_seed_raw_hashes=True, partial_reset_isolation=True,
                      new_seed_applied=True, pids_reused=True, learning_imports=forbidden)
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        try:
            backend.close()
        finally:
            report['seconds'] = time.monotonic() - started
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

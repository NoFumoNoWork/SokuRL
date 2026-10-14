"""Summarize fixed-grid evaluations and PPO diagnostics without model promotion."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_br_stage import evaluation_keys


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def win(game):
    return int(game['outcome'] == f"p{game['learner_seat'] + 1}_win")


def evaluate(directory):
    data = read(directory / 'result.json')
    if not data['success']:
        raise RuntimeError(f'incomplete evaluation: {directory}')
    games = data['result']['games']
    evaluation_keys(games)
    groups = {}
    for name in sorted({g['opponent'] for g in games}):
        rows = [g for g in games if g['opponent'] == name]
        received = sum(g['public_hp_loss_fraction_by_seat'][g['learner_seat']] for g in rows)
        dealt = sum(g['public_hp_loss_fraction_by_seat'][1-g['learner_seat']] for g in rows)
        groups[name] = dict(games=len(rows), wins=sum(map(win, rows)),
            timeouts=sum(g['outcome'] == 'time_limit' for g in rows),
            win_rate=float(np.mean([win(g) for g in rows])),
            mean_frames=float(np.mean([g['frames'] for g in rows])),
            public_damage_received=received / len(rows), public_damage_dealt=dealt / len(rows),
            pooled_public_damage_exchange=dealt / received if received else None,
            mean_base_reward=float(np.mean([g['reward_components_by_seat'][g['learner_seat']]['base'] for g in rows])),
            mean_shaping_reward=float(np.mean([g['reward_components_by_seat'][g['learner_seat']]['shaping'] for g in rows])),
            opponent_fingerprints=sorted({g['strategy_ids'][1-g['learner_seat']] for g in rows}))
    return data, games, groups


def paired(before, after):
    if evaluation_keys(before) != evaluation_keys(after):
        raise RuntimeError('paired trial identities differ')
    key = lambda g: (g['opponent'], g['world_seed'], g['learner_seat'])
    old = {key(g): g for g in before}
    result = {}
    for name in sorted({g['opponent'] for g in after}):
        blocks = {}
        for game in after:
            if game['opponent'] != name:
                continue
            previous = old[key(game)]
            if previous['strategy_ids'][1-previous['learner_seat']] != game['strategy_ids'][1-game['learner_seat']]:
                raise RuntimeError('opponent version differs')
            blocks.setdefault(game['world_seed'], []).append(win(game) - win(previous))
        differences = np.asarray([np.mean(row) for row in blocks.values()])
        if any(len(row) != 2 for row in blocks.values()):
            raise RuntimeError('both seats required per world seed')
        rng = np.random.default_rng(20261014)
        boot = differences[rng.integers(0, len(differences), size=(10000, len(differences)))].mean(axis=1)
        result[name] = dict(world_seed_blocks=len(differences),
            paired_win_rate_delta=float(differences.mean()),
            descriptive_block_bootstrap_95_interval=np.quantile(boot, [.025, .975]).tolist())
    return result


def diagnostics(directory):
    path = directory / 'scalars/progress.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for name in ('approx_kl', 'clip_fraction', 'entropy_loss', 'explained_variance', 'value_loss'):
        values = [float(row['train/'+name]) for row in rows if row.get('train/'+name)]
        result[name] = dict(recorded=len(values), mean=float(np.mean(values)),
            minimum=float(np.min(values)), maximum=float(np.max(values))) if values else dict(recorded=0)
    timing = read(directory / 'timing.json')['rollouts']
    result['timing'] = dict(rollout_seconds=sum(r['rollout_seconds'] for r in timing),
                          update_seconds=sum(r['update_seconds'] for r in timing))
    result['checkpoint'] = read(directory / 'result.json')['result']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline, old, groups = evaluate(args.baseline or args.runs / 'evaluation/ppo-fixed-baseline-20261014')
    report = dict(status='PASS', baseline=groups, arms={},
        interpretation='PASS means complete comparable evidence, not learning success; one training seed only',
        uncertainty='Fixed development grid, paired world-seed block bootstrap; not independent game Bernoulli CI')
    initial_hashes = set()
    for arm in ('a', 'b', 'c'):
        training = args.runs / f'ppo-controlled-{arm}-20261014'
        data, new, current = evaluate(args.runs / f'evaluation/ppo-controlled-{arm}-final-20261014')
        if baseline['runtime']['fingerprints']['game_id'] != data['runtime']['fingerprints']['game_id']:
            raise RuntimeError('original runtime identity differs')
        stats = diagnostics(training)
        initial_hashes.add(stats['checkpoint']['initial_policy_hash'])
        report['arms'][arm] = dict(evaluation=current, paired=paired(old, new), ppo=stats)
        interim = {}
        for steps in (65536, 131072):
            _, small, measured = evaluate(args.runs / f'evaluation/ppo-controlled-{arm}-{steps}-20261014')
            identities = {(g['opponent'], g['world_seed'], g['learner_seat']) for g in small}
            reference = [g for g in old if (g['opponent'], g['world_seed'], g['learner_seat']) in identities]
            interim[str(steps)] = dict(evaluation=measured, paired=paired(reference, small),
                scope='Eight-game pressure development subset; not comparable to full-grid rates directly')
        report['arms'][arm]['interim'] = interim
    if len(initial_hashes) != 1:
        raise RuntimeError('controlled arms did not start with identical parameters')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: comparable evidence; inspect per-opponent intervals before any strength claim')


if __name__ == '__main__':
    main()

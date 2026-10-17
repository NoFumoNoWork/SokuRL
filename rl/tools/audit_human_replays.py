"""Read-only source/contract audit before any human-replay supervision.

This tool does not infer replay poll indices from Bridge frame numbers, export
training labels, play the game, or inspect the sealed test set's input labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

from omegaconf import OmegaConf

from sokurl_platform.replay_format import Replay

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True)


def players(row):
    return {str(p['id']) for p in row['players'].values() if p.get('id') is not None}


def focal(row):
    return str(row['players'][row['sampling']['focal_side']]['id'])


def split_training(rows, seed):
    """Reserve validation players, removing cross-group opponents and SETs.

    The existing test assignment is immutable. No frame-level random split.
    """
    groups = sorted({focal(r) for r in rows})
    random.Random(seed).shuffle(groups)
    selected = set(groups[:max(1, len(groups) // 5)])
    validation = [r for r in rows if focal(r) in selected]
    validation_players = set().union(*(players(r) for r in validation))
    validation_sets = {r['set_id'] for r in validation}
    train = [r for r in rows if focal(r) not in selected
             and not (players(r) & validation_players) and r['set_id'] not in validation_sets]
    included = {r['id'] for r in train + validation}
    excluded = [r for r in rows if r['id'] not in included]
    return train, validation, excluded


def concentration(rows):
    per_player = Counter(p for r in rows for p in players(r))
    per_focal = Counter(focal(r) for r in rows)
    per_set = Counter(r['set_id'] for r in rows)
    return dict(games=len(rows), players=dict(per_player), marisa_players=dict(per_focal),
                sets=dict(per_set), max_player_games=max(per_player.values(), default=0),
                max_marisa_player_games=max(per_focal.values(), default=0),
                max_set_games=max(per_set.values(), default=0))


def choose_pilot(rows, count=20):
    chosen, remaining = [], sorted(rows, key=lambda r: str(r['id']))
    used_players, used_seats, used_months, used_sets = set(), set(), set(), set()
    while remaining and len(chosen) < count:
        def score(r):
            s = r['sampling']
            return (focal(r) not in used_players, s['focal_side'] not in used_seats,
                    s['month'] not in used_months, r['set_id'] not in used_sets)
        row = max(remaining, key=score)
        remaining.remove(row)
        chosen.append(row)
        used_players.add(focal(row))
        used_seats.add(row['sampling']['focal_side'])
        used_months.add(row['sampling']['month'])
        used_sets.add(row['set_id'])
    return chosen


def audit(output, index_path, checkpoint, seed):
    started = time.perf_counter()
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('output must be under rl/runs')
    output.mkdir(parents=True, exist_ok=False)
    index = json.loads(index_path.read_text(encoding='utf-8'))
    rows = [r for r in index['games'].values() if r['status'] == 'ok']
    inventory, failures, hashes = [], [], Counter()
    for path in sorted((ROOT.parent / 'third_party/SokuReplay').glob('*.rep')):
        digest = sha256(path)
        hashes[digest] += 1
        inventory.append(dict(id=path.stem, path=str(path), bytes=path.stat().st_size, sha256=digest))
    by_id = {r['id']: r for r in inventory}
    opponents, metadata = Counter(), Counter()
    verified = []
    for row in rows:
        try:
            item = by_id[str(row['id'])]
            if item['sha256'] != row['sha256'] or item['bytes'] != row['bytes']:
                raise ValueError('index/file byte identity differs')
            data = Path(item['path']).read_bytes()
            replay = Replay.decode(data)
            if replay.encode() != data:
                raise ValueError('replay byte roundtrip differs')
            side = 0 if row['sampling']['focal_side'] == 'server' else 1
            expected = tuple(row['players'][s]['character_id'] for s in ('server', 'client'))
            if any(tuple(p.character for p in m.players) != expected for m in replay.matches):
                raise ValueError('index and REP characters/seats differ')
            if expected[side] != 1:
                raise ValueError('focal player is not Marisa')
            opponents[expected[1-side]] += 1
            p = row['players'][row['sampling']['focal_side']]
            for key in ('id', 'league', 'elo', 'elo_before', 'elo_after'):
                metadata[key] += p.get(key) is not None
            verified.append(row)
        except (KeyError, ValueError, OSError) as error:
            failures.append(dict(id=row['id'], reason=str(error)))
    train_pool = [r for r in verified if r['sampling']['split'] == 'train']
    test = [r for r in verified if r['sampling']['split'] == 'test']
    reimu_pool = [r for r in train_pool if sorted(p['character_id'] for p in r['players'].values()) == [0, 1]]
    train, validation, excluded = split_training(reimu_pool, seed)
    split_rows = dict(train=train, validation=validation, test=test)
    overlap = {}
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        left, right = split_rows[a], split_rows[b]
        overlap[a+'_'+b] = dict(
            ids=sorted({r['id'] for r in left} & {r['id'] for r in right}),
            players=sorted(set().union(*(players(r) for r in left)) & set().union(*(players(r) for r in right))),
            sets=sorted({r['set_id'] for r in left} & {r['set_id'] for r in right}),
            sha256=sorted({r['sha256'] for r in left} & {r['sha256'] for r in right}))
    if any(any(value.values()) for value in overlap.values()):
        failures.append(dict(reason='split overlap', details=overlap))
    pilot = choose_pilot(train)
    config_path = checkpoint.parent / 'config.yaml'
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    write_new(output / 'inventory.json', inventory)
    write_new(output / 'splits.json', dict(schema='human_replay_split_v1', seed=seed,
        source_index=str(index_path), source_sha256=sha256(index_path),
        method='Preserve test; hold out focal Marisa players from original Reimu training pool; remove cross-player/SET edges',
        test_labels_for_selection=False, **{s: r for s, r in split_rows.items()}, excluded=excluded))
    write_new(output / 'pilot.json', dict(schema='human_replay_pilot_v1', rows=pilot,
        status='PLANNED_NOT_SUPERVISED', expected_count=20))
    from stable_baselines3 import PPO
    from soku_rl.env.hisouten_env import EpisodeConfig
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
    from soku_rl.policy.contract import read_training_contract
    from soku_rl.rl.ppo import parameter_hash
    model = PPO.load(checkpoint, device='cpu')
    interface = LearningInterface(EpisodeConfig.from_dict(config['episode']), LearningConfig(**config['wrappers']))
    read_training_contract(config_path, interface)
    if model.action_space != interface.action_space or model.observation_space != interface.observation_space:
        raise ValueError('checkpoint/interface spaces differ')
    contract = dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint),
        config=str(config_path), config_sha256=sha256(config_path),
        loaded_parameter_sha256=parameter_hash(model.policy), policy_class=type(model.policy).__name__,
        extractor_class=type(model.policy.features_extractor).__name__,
        observation_shape=[int(n) for n in model.observation_space.shape], actions=int(model.action_space.n),
        masked_actions=list(model.policy.masked_actions),
        episode=config['episode'], wrappers=config['wrappers'], ppo=config['rl']['ppo'],
        old_runtime_paths=config['runtime'], execution_note='Historical runtime paths retained as evidence, never executed',
        status='PASS', native_execution='NOT_TESTED', optimizer='NOT_CREATED_NO_TRAINING')
    write_new(output / 'model_contract.json', contract)
    result = dict(schema='human_replay_audit_v1', status='FAIL' if failures else 'PASS',
        inventory_files=len(inventory), unique_ids=len(by_id), unique_sha256=len(hashes),
        duplicate_sha256={h: n for h, n in hashes.items() if n > 1},
        indexed_success=len(rows), indexed_verified=len(verified), failures=failures,
        opponent_character_counts=dict(opponents), focal_metadata_available=dict(metadata),
        historical_rating_time='UNVERIFIED; site collection ratings are not proven match-time ratings',
        original_train=concentration(train_pool), original_test=concentration(test),
        original_reimu_train=len(reimu_pool),
        splits={s: concentration(r) for s, r in split_rows.items()}, overlap=overlap,
        excluded_cross_group=len(excluded), pilot_count=len(pilot),
        pilot_distinct_marisa_players=len({focal(r) for r in pilot}),
        pilot_seats=dict(Counter(r['sampling']['focal_side'] for r in pilot)),
        pilot_months=dict(Counter(r['sampling']['month'] for r in pilot)),
        native_alignment='NOT_TESTED', compatible_segments='NOT_TESTED',
        valid_supervised_frames=0, training='NOT_STARTED', elapsed_seconds=time.perf_counter()-started)
    write_new(output / 'audit.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'inventory_files', 'indexed_verified',
        'original_reimu_train', 'excluded_cross_group', 'pilot_count', 'pilot_seats', 'elapsed_seconds')}))
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=Path, default=ROOT.parent / 'platform/reports/sokureplays-priority-20261016/new1000/index.json')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/combat-v2-adaptive-20261015/campaign/train-4/final.zip')
    parser.add_argument('--seed', type=int, default=20261016)
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.output, args.index, args.checkpoint, args.seed) else 1)

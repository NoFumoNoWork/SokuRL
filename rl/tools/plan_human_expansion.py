"""Extend a frozen REP training split, never reassign validation/test players."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json

from audit_human_replays import choose_pilot, players
from sokurl_platform.replay_format import Replay
from soku_rl.replay.human_dataset import save_json, sha256


def candidates(index, splits):
    reserved = splits['validation'] + splits['test']
    old = splits['train'] + reserved + splits['excluded']
    ids = {str(r['id']) for r in old}
    hashes = {r['sha256'] for r in old}
    held_players = {p for r in reserved for p in players(r)}
    held_sets = {str(r['set_id']) for r in reserved}
    result, exclusions = [], Counter()
    for source in sorted(index['games'].values(), key=lambda r: int(r['id'])):
        reason = None
        if source['status'] != 'ok':
            reason = 'SOURCE_NOT_OK'
        elif str(source['id']) in ids or source['sha256'] in hashes:
            reason = 'ALREADY_ASSIGNED_OR_DUPLICATE'
        elif sorted(p['character_id'] for p in source['players'].values()) != [0, 1]:
            reason = 'OTHER_MATCHUP'
        elif (source.get('set_id') is None or
              any(p.get('id') is None for p in source['players'].values())):
            reason = 'UNKNOWN_PLAYER_OR_SET'
        elif (players(source) & held_players or str(source['set_id']) in held_sets
              or source.get('sampling', {}).get('split') == 'test'):
            reason = 'RESERVED_PLAYER_OR_SET'
        if reason:
            exclusions[reason] += 1
            continue
        row = deepcopy(source)
        side = next(s for s, p in row['players'].items() if p['character_id'] == 1)
        row['sampling'] = dict(split='train', focal_side=side, month=row['time_utc'][:7],
                               source='FIXED_PARTITION_EXTENSION_NO_TEST_SELECTION')
        result.append(row)
        ids.add(str(row['id']))
        hashes.add(row['sha256'])
    return result, dict(exclusions)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--index', type=Path, default=Path('third_party/SokuReplay/index.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-new', type=int, default=32)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(root/'rl/runs') or not 1 <= args.max_new <= 200:
        raise ValueError('new local output and bounded expansion required')
    if json.loads((args.audit/'audit.json').read_text())['status'] != 'PASS':
        raise ValueError('previous audit must pass')
    split_file = args.audit/'splits.json'
    original = json.loads(split_file.read_text())
    index = json.loads(args.index.read_text())
    pool, excluded = candidates(index, original)
    selected = choose_pilot(pool, min(args.max_new, len(pool)))
    if not selected:
        raise ValueError('BLOCKED_NO_NEW_DISJOINT_TRAINING_REP')
    for row in selected:
        path = Path(row['file']).resolve(strict=True)
        if not path.is_relative_to(root/'third_party/SokuReplay'):
            raise ValueError('source outside shared REP directory')
        data = path.read_bytes()
        if sha256(path) != row['sha256'] or len(data) != row['bytes']:
            raise ValueError('indexed source byte identity differs')
        replay = Replay.decode(data)
        expected = tuple(row['players'][s]['character_id'] for s in ('server', 'client'))
        if replay.encode() != data or not replay.matches or any(
                tuple(p.character for p in m.players) != expected for m in replay.matches):
            raise ValueError('REP roundtrip or character/seat identity differs')
    out.mkdir(parents=True)
    splits = deepcopy(original)
    splits['train'] += selected
    splits.update(parent_split=str(split_file.resolve()), parent_split_sha256=sha256(split_file),
        extension_index=str(args.index.resolve()), extension_index_sha256=sha256(args.index),
        method='Append audited local training only;freeze all old partitions and exclusions')
    save_json(out/'splits.json', splits)
    for part in ('validation', 'test', 'excluded'):
        if splits[part] != original[part]:
            raise AssertionError('reserved partition changed')
    save_json(out/'new-training-plan.json', dict(schema='human_replay_export_plan_v1',
        partition='train', split_sha256=sha256(out/'splits.json'), rows=selected))
    report = dict(status='PASS', scope='STATIC_SOURCE_IDENTITY_AND_FIXED_PARTITION_EXTENSION',
        previous_audit_sha256=sha256(args.audit/'audit.json'),
        source_split_sha256=sha256(split_file), source_index_sha256=sha256(args.index),
        old_training=len(original['train']), added_training=len(selected), training=len(splits['train']),
        candidates=len(pool), exclusions=excluded, validation_reserved=len(original['validation']),
        sealed_test=len(original['test']), reserved_partitions_exact=True,
        native_alignment='NOT TESTED_NEW_CASES', training_execution='NOT RUN',
        sealed_test_labels_used=False)
    save_json(out/'audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

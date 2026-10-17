"""Augment accepted v1 event rows without altering source labels or raw readers."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from soku_rl.replay.hierarchical import LATEST
from soku_rl.replay.semantic_events import SemanticEventHistory, contact_feedback, event_strata
from soku_rl.replay.human_dataset import sha256, save_json
from soku_rl.rl.hierarchical_data import EventDataset
from sokurl_platform.replay_inputs import decode_keys


def checked(path, digest):
    path = Path(path).resolve(strict=True)
    if sha256(path) != digest:
        raise ValueError(f'changed source: {path}')
    return path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(root / 'rl/runs'):
        raise ValueError('new project-local output required')
    # Validate existing case hashes and forbid sealed-test partitions.
    EventDataset(args.data, 'train'); EventDataset(args.data, 'validation')
    old = json.loads(args.data.read_text())
    chain_path = checked(old['source'], old['source_sha256'])
    chain_manifest = json.loads(chain_path.read_text())
    lookup = {(c['partition'], c['replay_id']): c for c in chain_manifest['cases']}
    out.mkdir(parents=True)
    manifest = dict(format='semantic_event_dataset_v2', status='RUNNING', catalog=old['catalog'],
        source=str(args.data.resolve()), source_sha256=sha256(args.data), cases=[],
        evidence_boundary=dict(contact='CONFIRMED_PLATFORM_HIT_BLOCK',
            link='OBSERVED_SUCCESSION_NOT_PROVEN_CANCEL', whiff='UNKNOWN', cancel='UNKNOWN', recovery='UNKNOWN',
            masks='CANDIDATE_VOCAB_NOT_ENGINE_LEGAL_SETS', future_inputs='FORBIDDEN', sealed_test_used=False))
    save_json(out / 'manifest.json', manifest)
    try:
        for case in old['cases']:
            part = case['partition']
            if part not in ('train', 'validation'):
                raise ValueError('sealed test forbidden')
            evidence = lookup[part, case['replay_id']]
            chain_folder = chain_path.parent / part / str(case['replay_id'])
            chains_file = checked(chain_folder / 'chains.json', case['chains_sha256'])
            chains = json.loads(chains_file.read_text())
            alignment_file = args.data.parent / part / str(case['replay_id']) / 'alignment.json'
            audit = json.loads(alignment_file.read_text())
            source_manifest = checked(case['source_manifest'], case['source_manifest_sha256'])
            original = json.loads(source_manifest.read_text())
            item = next(c for c in original['replays'] if c['replay_id'] == case['replay_id'])
            folder = source_manifest.parent / str(case['replay_id'])
            events_file = checked(folder / 'events.jsonl.gz', item['events_sha256'])
            with gzip.open(events_file, 'rt', encoding='utf-8') as stream:
                records = [json.loads(line) for line in stream]
            parts = []
            for shard in item['shards']:
                with np.load(checked(folder / shard['path'], shard['sha256']), allow_pickle=False) as z:
                    parts.append({k: z[k] for k in z.files})
            arrays = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
            with np.load(checked(args.data.parent / case['path'], case['sha256']), allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            if (len(audit['samples']) != len(data['state']) or len(records) != len(arrays['actions'])
                    or not np.array_equal(arrays['consumption_frames'], arrays['observation_frames'] + 1)):
                raise ValueError('source row alignment mismatch')
            expected = {int(f): [] for f in arrays['observation_frames']}
            incoming = set()
            feedback_count = dict(HIT=0, BLOCK=0)
            for record in records:
                frame = record['frame']
                feedback = contact_feedback(record['native_events'], item['seat'], frame)
                expected[frame] = feedback
                for event in feedback:
                    feedback_count[event['kind']] += 1
                    if event['owner'] == 1:
                        incoming.add(frame)
            rows = {s['row']: i for i, s in enumerate(audit['samples'])}
            valid = arrays['compatible'] & ~arrays['boundary'] & arrays['raw_words_present']
            history = SemanticEventHistory()
            for row in range(len(valid)):
                if not valid[row]:
                    history.reset()
                    continue
                frame = int(arrays['observation_frames'][row])
                if int(records[row]['frame']) != int(arrays['consumption_frames'][row]):
                    raise ValueError('native events and state times differ')
                previous = (tuple(decode_keys(int(arrays['raw_words'][row-1, item['seat']])).values())
                            if row and valid[row-1] else None)
                history.observe(arrays['observations'][row, LATEST], frame, previous,
                                feedback=expected.get(frame, ()))
                if row in rows:
                    i = rows[row]
                    if frame != audit['samples'][i]['observation_frame']:
                        raise ValueError('audit observation time mismatch')
                    features = history.tensors()
                    for key in ('state', 'inputs'):
                        if not np.array_equal(features[key], data[key][i]):
                            raise ValueError('state or own pre-action history changed')
                    for key in ('event_ids', 'event_states', 'event_numbers', 'event_mask'):
                        data[key][i] = features[key]
            strata, chain_ids = event_strata(data, audit, chains, incoming)
            data.update(supervision_strata=strata, replay_id=np.full(len(strata), case['replay_id'], np.int64),
                        chain_id=chain_ids)
            destination = out / part / str(case['replay_id'])
            destination.mkdir(parents=True)
            np.savez_compressed(destination / 'samples.npz', **data)
            save_json(destination / 'alignment.json', audit)
            manifest['cases'].append(dict(partition=part, replay_id=case['replay_id'], samples=len(strata),
                source_shard_sha256=case['sha256'], alignment_sha256=sha256(alignment_file),
                events_sha256=sha256(events_file), chains_sha256=sha256(chains_file),
                path=str((destination / 'samples.npz').relative_to(out)), sha256=sha256(destination / 'samples.npz'),
                strata=strata.sum(0).tolist(), feedback=feedback_count))
            save_json(out / 'manifest.json', manifest)
        manifest['status'] = 'PASS_CAUSAL_DERIVATIVE_LIMITED_FEEDBACK'
        manifest['totals'] = {part: dict(cases=sum(c['partition'] == part for c in manifest['cases']),
            samples=sum(c['samples'] for c in manifest['cases'] if c['partition'] == part),
            strata=np.sum([c['strata'] for c in manifest['cases'] if c['partition'] == part], axis=0).tolist(),
            feedback={kind: sum(c['feedback'][kind] for c in manifest['cases'] if c['partition'] == part)
                      for kind in ('HIT', 'BLOCK')}) for part in ('train', 'validation')}
    except BaseException as error:
        manifest.update(status='FAIL', error=repr(error))
        raise
    finally:
        save_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest['totals'], indent=2))


if __name__ == '__main__':
    main()

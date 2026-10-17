"""Derive post-300F training eligibility; preserve every source array and row."""
import argparse
import json
from pathlib import Path

import numpy as np

from soku_rl.replay.combat_windows import post_opening_selection, SELECTION_REASONS
from soku_rl.replay.human_dataset import sha256, save_json
from soku_rl.rl.hierarchical_data import SemanticDataset


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--min-frame', type=int, default=300)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(root/'rl/runs') or args.min_frame < 0:
        raise ValueError('new local output and nonnegative cutoff required')
    source = args.data.resolve(strict=True)
    SemanticDataset(source,'train'); SemanticDataset(source,'validation')
    original = json.loads(source.read_text())
    out.mkdir(parents=True)
    manifest = dict(format='semantic_post_opening_dataset_v3',status='RUNNING',catalog=original['catalog'],
        source=str(source),source_sha256=sha256(source),cases=[],
        selection=dict(min_observation_frame=args.min_frame,reason_names=SELECTION_REASONS,
            object_source='CURRENT_PUBLIC_VISIBLE_OBJECT_TOKENS_16;NOT_ATTACKABILITY',
            own_attack_required_for_mobility=False,original_rows_and_history_preserved=True,
            all_reliable_post_opening_attacks=True,sealed_test_used=False),
        evaluation='Validation REP/split/labels unchanged;full natural validation remains default.')
    save_json(out/'manifest.json',manifest)
    try:
        for case in original['cases']:
            if case['partition'] not in ('train','validation'):
                raise ValueError('sealed test forbidden')
            path = source.parent/case['path']
            if sha256(path) != case['sha256']:
                raise ValueError('source shard changed')
            with np.load(path,allow_pickle=False) as z:
                data = {k:z[k] for k in z.files}
            alignment_path = path.parent/'alignment.json'
            audit = json.loads(alignment_path.read_text())
            selection, stats = post_opening_selection(data,audit,original['catalog'],args.min_frame)
            if set(selection) & set(data):
                raise ValueError('do not overwrite original arrays')
            target = out/case['partition']/str(case['replay_id'])
            target.mkdir(parents=True)
            np.savez_compressed(target/'samples.npz',**data,**selection)
            save_json(target/'alignment.json',audit)
            # Independent readback, not merely a successful write.
            with np.load(target/'samples.npz',allow_pickle=False) as z:
                for key,value in data.items():
                    if z[key].dtype != value.dtype or not np.array_equal(z[key],value):
                        raise ValueError('original state/input/target/history changed')
            manifest['cases'].append(dict(partition=case['partition'],replay_id=case['replay_id'],
                samples=len(data['state']),path=str((target/'samples.npz').relative_to(out)),
                sha256=sha256(target/'samples.npz'),source_shard_sha256=case['sha256'],
                source_alignment_sha256=sha256(alignment_path),**stats))
            save_json(out/'manifest.json',manifest)
        manifest['totals'] = {}
        for part in ('train','validation'):
            cases = [c for c in manifest['cases'] if c['partition']==part]
            keys = [k for k in cases[0] if k in stats and k != 'reasons']
            manifest['totals'][part] = dict(cases=len(cases),
                **{k:sum(c[k] for c in cases) for k in keys},
                reasons={k:sum(c['reasons'][k] for c in cases) for k in SELECTION_REASONS})
        if manifest['totals']['train']['eligible_rows']==0:
            raise ValueError('no eligible training data;do not fall back to opening frames')
        manifest['status'] = 'PASS_OFFLINE_SELECTION_ORIGINAL_ARRAYS_EXACT'
    except BaseException as error:
        manifest.update(status='FAIL',error=repr(error))
        raise
    finally:
        save_json(out/'manifest.json',manifest)
    print(json.dumps(manifest['totals'],indent=2))


if __name__=='__main__':
    main()

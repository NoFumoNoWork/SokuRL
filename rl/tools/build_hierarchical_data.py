"""Offline event/onset dataset, preserving evidence masks and original conditions."""
import argparse
import json
from pathlib import Path

import numpy as np

from soku_rl.replay.hierarchical import common_bindings,build_hierarchical_case
from soku_rl.replay.human_dataset import sha256,save_json
from soku_rl.policy.hierarchical_control import command_catalog


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--chains',type=Path,required=True)
    parser.add_argument('--motion',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];out=args.output.resolve()
    if out.exists() or not out.is_relative_to(root/'rl/runs'):raise ValueError('new local runs output required')
    chain_manifest=args.chains/'manifest.json';source=json.loads(chain_manifest.read_text())
    if source['status']!='PASS_OFFLINE_EXTRACTION_NOT_POLICY_VALIDATION':raise ValueError('unaccepted chain source')
    bindings=common_bindings(args.motion)
    out.mkdir(parents=True)
    manifest=dict(format='hierarchical_dataset_v1',status='RUNNING',catalog=command_catalog(),cases=[],
        source=str(chain_manifest.resolve()),source_sha256=sha256(chain_manifest),
        motion=str(args.motion.resolve()),motion_sha256=sha256(args.motion),bindings=bindings,
        contract=dict(state=[185],event_states=[16,185],event_ids=[16,3],event_numbers=[16,3],
            event_mask=[16],inputs=[60,8],sequence_targets=[3]),
        limitations=['Only unique raw edges joined to community action and native acceptance label attacks.',
            'No proof of latent intention or exact buffered key identity.',
            'Unmapped skills/charge/ambiguous onsets are masked,not labelled as neutral.',
            'Candidate masks are NOT complete native legal action sets.',
            'Public events only;native contact/Combo outcomes never enter policy inputs.',
            'Termination supervises observed switching/continuation at public events,not forced transitions.',
            'No native legality release-time proof;no policy strength/PPO compatibility claim.'])
    save_json(out/'manifest.json',manifest)
    try:
        for case in source['cases']:
            folder=args.chains/case['partition']/str(case['replay_id'])
            for name,digest in case['artifacts'].items():
                if sha256(folder/name)!=digest:raise ValueError('chain artifact changed')
            p=Path(case['source_manifest'])
            if sha256(p)!=case['source_manifest_sha256']:raise ValueError('source manifest changed')
            original=json.loads(p.read_text());item=next(c for c in original['replays'] if c['replay_id']==case['replay_id'])
            parts=[]
            for shard in item['shards']:
                path=p.parent/str(case['replay_id'])/shard['path']
                if sha256(path)!=shard['sha256']:raise ValueError('source shard changed')
                with np.load(path,allow_pickle=False) as z:parts.append({k:z[k] for k in z.files})
            arrays={k:np.concatenate([d[k] for d in parts]) for k in parts[0]}
            chains=json.loads((folder/'chains.json').read_text())
            data,audit=build_hierarchical_case(arrays,chains,bindings)
            destination=out/case['partition']/str(case['replay_id']);destination.mkdir(parents=True)
            np.savez_compressed(destination/'samples.npz',**data)
            save_json(destination/'alignment.json',audit)
            counts=dict(samples=len(audit['samples']),action_labels=int(data['target_action_mask'].sum()) if data else 0,
                termination_labels=int(data['target_termination_mask'].sum()) if data else 0,
                sequence_labels=int(data['target_sequence_mask'].sum()) if data else 0,
                unknown_onsets=len(audit['unknown']))
            manifest['cases'].append(dict(partition=case['partition'],replay_id=case['replay_id'],**counts,
                source_manifest=str(p),source_manifest_sha256=sha256(p),
                chains_sha256=sha256(folder/'chains.json'),
                path=str((destination/'samples.npz').relative_to(out)),sha256=sha256(destination/'samples.npz')))
            save_json(out/'manifest.json',manifest)
        manifest['totals']={part:{key:sum(c[key] for c in manifest['cases'] if c['partition']==part)
            for key in ('samples','action_labels','termination_labels','sequence_labels','unknown_onsets')}
            for part in ('train','validation')}
        manifest['status']='PASS_OFFLINE_CAUSAL_LABELS_LIMITED_LEGALITY'
    except BaseException as error:
        manifest.update(status='FAIL',error=repr(error));raise
    finally:save_json(out/'manifest.json',manifest)
    print(json.dumps(manifest['totals'],indent=2))


if __name__=='__main__':main()

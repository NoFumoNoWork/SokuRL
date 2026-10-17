"""Plan more audited train REP,and reference existing accepted shards unchanged."""
import argparse
import json
from pathlib import Path

from audit_human_replays import choose_pilot
from soku_rl.replay.human_dataset import sha256,save_json
from soku_rl.rl.human_cloning import HumanCollection


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path,default=Path('rl/runs/human-bc-20261016/audit-v3'))
    parser.add_argument('--previous-data',type=Path,default=Path('rl/runs/human-bc-export-20261017'))
    parser.add_argument('--warm',type=Path,default=Path('rl/runs/human-bc-c-weighted-20261017/C/best.json'))
    parser.add_argument('--total',type=int,default=60)
    parser.add_argument('--finalize',action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    if not out.is_relative_to(root/'rl/runs') or not 21<=args.total<=100:
        raise ValueError('bounded new project-local scale-up required')
    if args.finalize:
        if (out/'train-collection.json').exists(): raise FileExistsError('preserve existing collection')
        plan = json.loads((out/'scale-plan.json').read_text())
        items = []
        for manifest,acceptance in [(args.previous_data/'pilot-20/manifest.json',args.previous_data/'pilot-acceptance-v2.json'),
                                   (out/'new-training/manifest.json',out/'new-training-acceptance.json')]:
            items.append({key:str(p.resolve(strict=True)) for key,p in [('manifest',manifest),('acceptance',acceptance)]})
            items[-1].update({key+'_sha256':sha256(value) for key,value in list(items[-1].items())})
        save_json(out/'train-collection.json',dict(schema='human_replay_collection_v1',datasets=items,
            note='References accepted originals;no shard copies,history splicing or synthetic acceptance.'))
        collection = HumanCollection(out/'train-collection.json')
        if [c['replay_id'] for c in collection.manifest['replays']]!=plan['selected_ids']:
            raise ValueError('accepted collection does not cover exact planned training set')
        print('PASS accepted training collection:',len(collection.manifest['replays']))
        return
    if out.exists(): raise FileExistsError('preserve previous scale-up')
    splits_path = args.audit/'splits.json'
    splits = json.loads(splits_path.read_text())
    selected = choose_pilot(splits['train'],args.total)
    previous = json.loads((args.previous_data/'pilot-20/manifest.json').read_text())
    old_ids = [c['replay_id'] for c in previous['replays']]
    if [r['id'] for r in selected[:20]]!=old_ids or len(selected)!=args.total:
        raise ValueError('deterministic prefix does not preserve existing20 sources')
    out.mkdir(parents=True)
    save_json(out/'new-training-plan.json',dict(schema='human_replay_export_plan_v1',partition='train',
        split_sha256=sha256(splits_path),rows=selected[20:]))
    save_json(out/'scale-plan.json',dict(selected_ids=[r['id'] for r in selected],
        reused_ids=old_ids,new_ids=[r['id'] for r in selected[20:]],
        split_source=str(splits_path.resolve()),split_sha256=sha256(splits_path),
        training_total=args.total,validation_reused=20,epochs=20,max_anchor_visits=2000000,
        original_conditions='unchanged;only existing compatible prefixes accepted',sealed_test_used=False))
    save_json(out/'model_alias.json',dict(name='D',definition='Previously named weighted C;L0+1/4*sum(L1..L4)',
        source_manifest=str(args.warm.resolve(strict=True)),source_sha256=sha256(args.warm),
        action_interface='90actions,1F',source_unchanged=True,
        warmstart='Full actor/context/future weights;fresh BC optimizer,not restored moments.'))
    from game_runtime.identity import fingerprints
    save_json(out/'before.json',dict(runtime=fingerprints(),
        capture_adapter_sha256=sha256(root/'rl/src/soku_rl/replay/human_dataset.py')))
    print('Planned:',len(selected),'training REP;reuse20,export',len(selected)-20)


if __name__=='__main__':
    main()

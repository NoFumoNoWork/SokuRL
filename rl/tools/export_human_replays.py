"""Export the audited human training pilot using existing native replay APIs."""
import argparse
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from omegaconf import OmegaConf
from stable_baselines3 import PPO
import torch

from soku_rl.replay.human_dataset import export_replay, save_json, sha256
from soku_rl.replay.human_plans import export_plan


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=7200)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prefix-only', action='store_true')
    parser.add_argument('--partition', choices=('train','validation'), default='train')
    parser.add_argument('--plan',type=Path,help='Explicit subset bound to audited split;capture implementation unchanged')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not output.is_relative_to(root/'rl/runs') or not 1 <= args.frames <= 24000 or not 1 <= args.limit <= 200:
        raise ValueError('bounded output must be under rl/runs,1..200 records,1..24000 frames')
    if output.exists() and not args.resume:
        raise FileExistsError('preserve previous output; use explicit resume only for completed shards')
    output.mkdir(parents=True, exist_ok=args.resume)
    plan_path,plan = export_plan(args.audit,args.partition,args.plan)
    if args.limit>len(plan['rows']): raise ValueError('limit exceeds planned rows')
    config = OmegaConf.to_container(OmegaConf.load(args.source/'config.yaml'), resolve=True)
    torch.set_num_threads(4)
    model = PPO.load(args.source/'final.zip', device='cpu')
    blocked = set(model.policy.masked_actions)
    identity = dict(plan_sha256=sha256(plan_path), source_checkpoint_sha256=sha256(args.source/'final.zip'),
        config_sha256=sha256(args.source/'config.yaml'), exporter_sha256=sha256(Path(__file__)),
        adapter_sha256=sha256(root/'rl/src/soku_rl/replay/human_dataset.py'), frames=args.frames,
        prefix_only=args.prefix_only,partition=args.partition)
    manifest_path = output/'manifest.json'
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            raise ValueError('resume identity changed')
    else:
        manifest = dict(schema='human_replay_dataset_v1', status='RUNNING', identity=identity,
            partition=args.partition, sealed_test_used=False, replays=[], native_changes=False,
            gate='EXPORT_NOT_FULL_PILOT_ACCEPTANCE', training=False)
        code = output/'code'
        code.mkdir()
        for path in (Path(__file__), root/'rl/src/soku_rl/replay/human_dataset.py'):
            (code/path.name).write_bytes(path.read_bytes())
    save_json(manifest_path, manifest)
    start = time.perf_counter()
    try:
        with patch.dict(os.environ, {'SOKURL_REPLAY_INPUT_CAPTURE':'0', 'SOKURL_COMBAT_EVENTS':'1',
                'SOKURL_NATIVE_TRACE':'0', 'SOKURL_NATIVE_INPUT_COUNTERS':'0', 'SOKURL_CARDS_DISABLED':'0',
                'SOKURL_FRESH_RESET':'0', 'SOKURL_HEADLESS_RENDER':'1', 'SOKURL_IDLE_WAIT':'1',
                'SOKURL_MUTE_AUDIO':'1', 'SOKURL_ORIGINAL_REPLAY_RECORDING':'0'}):
            for row in plan['rows'][:args.limit]:
                folder = output/str(row['id'])
                if folder.exists():
                    previous = next((r for r in manifest['replays'] if r['replay_id'] == row['id']), None)
                    if previous is None or previous['status'] != 'PASS_EXPORT':
                        raise ValueError('incomplete record is preserved; retry in a new run,not overwrite')
                    for shard in previous['shards']:
                        if sha256(folder/shard['path']) != shard['sha256']:
                            raise ValueError('resume shard hash changed')
                    if sha256(row['file']) != previous['source_sha256']:
                        raise ValueError('resume source changed')
                    if (sha256(folder/'states.raw.gz') != previous['raw_sha256']
                            or sha256(folder/'events.jsonl.gz') != previous['events_sha256']):
                        raise ValueError('resume native evidence changed')
                    continue
                result = export_replay(row, config, args.source/'config.yaml', blocked, folder,
                    args.frames, prefix_only=args.prefix_only)
                manifest['replays'].append(result)
                save_json(manifest_path, manifest)
                print(json.dumps({k:result[k] for k in ('replay_id','status','rows','supervised','stop_reason','seconds')}), flush=True)
                if result['status'] != 'PASS_EXPORT':
                    raise ValueError('core data gate failed; do not train')
        manifest['status'] = 'PASS_EXPORT_ONLY'
    except BaseException as error:
        manifest.update(status='BLOCKED', error=repr(error))
        raise
    finally:
        manifest['invocation_seconds'] = time.perf_counter()-start
        save_json(manifest_path, manifest)


if __name__ == '__main__':
    main()

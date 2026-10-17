"""Train accepted human shards from exact PPO weights with a new optimizer."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import zipfile

from omegaconf import OmegaConf
import torch

from soku_rl.replay.human_dataset import policy_interface, save_json, sha256
from soku_rl.rl.human_cloning import HumanDataset, fit_human


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--train-acceptance',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--validation-acceptance',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=20261017)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not output.is_relative_to(root/'rl/runs') or output.exists():
        raise ValueError('use a new project rl/runs output')
    train = HumanDataset(args.train,args.train_acceptance)
    validation = HumanDataset(args.validation,args.validation_acceptance)
    def identities(dataset):
        return ({r['replay_id'] for r in dataset.manifest['replays']},
                {r['source_sha256'] for r in dataset.manifest['replays']},
                {r['set_id'] for r in dataset.manifest['replays']})
    if any(a&b for a,b in zip(identities(train),identities(validation))):
        raise ValueError('train/validation identity overlap')
    if set(train.acceptance['players']) & set(validation.acceptance['players']):
        raise ValueError('human players overlap across training and validation')
    if train.manifest['identity']['source_checkpoint_sha256'] != validation.manifest['identity']['source_checkpoint_sha256']:
        raise ValueError('dataset checkpoint contracts differ')
    config = OmegaConf.to_container(OmegaConf.load(args.source/'config.yaml'),resolve=True)
    interface = policy_interface(config,(1,0))
    output.mkdir(parents=True)
    for name in ('temp','cuda-cache','torch-cache'):
        (output/name).mkdir()
    os.environ.update(TEMP=str(output/'temp'),TMP=str(output/'temp'),
        CUDA_CACHE_PATH=str(output/'cuda-cache'),TORCH_HOME=str(output/'torch-cache'))
    config['runtime']['cwd'] = str(root)
    config['runtime']['command'] = [sys.executable,'-B',str(root/'rl/tools/rollout_worker.py')]
    config['output'] = str(output)
    OmegaConf.save(OmegaConf.create(config),output/'config.yaml')
    save_json(output/'human_bc_config.json',dict(epochs=10,batch_size=128,value_coef=0,
        learning_rate=config['rl']['ppo']['learning_rate'],action_change_weight=1.0,
        change_weight_source='rl/config/pretrain_demonstrations.yaml',
        source_checkpoint_sha256=sha256(args.source/'final.zip'),
        train_manifest=str(args.train),validation_manifest=str(args.validation),
        train_acceptance=str(args.train_acceptance),validation_acceptance=str(args.validation_acceptance),
        code_hashes={str(p.relative_to(root)):sha256(p) for p in (
            Path(__file__),root/'rl/src/soku_rl/rl/human_cloning.py',root/'rl/src/soku_rl/rl/ppo.py')}))
    from game_runtime.identity import fingerprints
    identity = fingerprints()
    identity['hub_head'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    identity['hub_dirty'] = bool(subprocess.check_output(['git','status','--porcelain','-uno'],cwd=root,text=True).strip())
    identity['published_anchors'] = dict(platform='7bd1d6668984710dec1d5d8a4b883637a882a54e',
        RL='a5e634950df21300bb2888a74ebd4a4fd16aa4aa',
        note='Historical published anchors,not exact runtime SHAs;both module trees have later uncommitted changes.')
    with zipfile.ZipFile(output/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for part in ('rl/src','rl/tools','rl/config','platform/src','platform/tools','platform/native/SokuRLBridge'):
            for path in sorted((root/part).rglob('*')):
                if path.is_file() and path.suffix in ('.py','.ps1','.yaml','.cpp','.hpp','.h','.toml'):
                    archive.write(path,path.relative_to(root).as_posix())
    identity['source_archive_sha256'] = sha256(output/'source.zip')
    save_json(output/'identity.json',identity)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    fit_human(interface,config['rl'],dict(kind='weights',path=str((args.source/'final.zip').resolve()),
        training_config=str((args.source/'config.yaml').resolve())),train,validation,output,args.device,args.seed)


if __name__ == '__main__':
    main()

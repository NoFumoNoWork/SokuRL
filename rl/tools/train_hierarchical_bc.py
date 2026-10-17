"""Bounded supervised correctness entry;no implicit large training budget."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from soku_rl.rl.hierarchical_bc import HierarchicalActor,hierarchical_loss,save_hierarchical,load_hierarchical
from soku_rl.rl.hierarchical_data import EventDataset,tensors
from soku_rl.replay.human_dataset import sha256,save_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('rl/config/hierarchical_bc.yaml'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--training-config',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2];out=args.output.resolve()
    if args.training_config:
        if args.updates:raise ValueError('do not mix correctness and formal budgets')
        from soku_rl.rl.hierarchical_training import run_training
        return run_training(args.training_config,args.data,out,args.device,root)
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    if (out.exists() or not out.is_relative_to(root/'rl/runs') or
            not 0<=args.updates<=min(16,cfg['max_updates']) or cfg['train_long_runs']):
        raise ValueError('new local run and bounded explicit correctness budget required')
    if [cfg[k] for k in ('event_history','event_layers','event_width','event_heads','input_history','sequence_horizon')]!=[16,3,128,4,60,3]:
        raise ValueError('architecture contract differs')
    if cfg['termination_weight']!=.5 or cfg['sequence_weight']!=.5:raise ValueError('not the approved objective')
    out.mkdir(parents=True)
    for name in ('temp','cuda-cache','torch-cache'):(out/name).mkdir()
    os.environ.update(TEMP=str(out/'temp'),TMP=str(out/'temp'),CUDA_CACHE_PATH=str(out/'cuda-cache'),TORCH_HOME=str(out/'torch-cache'))
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    from stable_baselines3 import PPO
    source=Path(cfg['source_checkpoint']).resolve(strict=True)
    original=PPO.load(source,device='cpu')
    train=EventDataset(args.data,'train');validation=EventDataset(args.data,'validation')
    actor=HierarchicalActor(original.policy.features_extractor,len(train.manifest['catalog'])).to(args.device)
    optimizer=torch.optim.Adam(actor.parameters(),lr=cfg['learning_rate'])
    batch_size=cfg['batch_size'];report=dict(status='RUNNING',updates=0,history=[],source=str(source),source_sha256=sha256(source),
        data=str(args.data.resolve()),data_sha256=sha256(args.data),config=cfg,
        value_supervision=False,policy_improvement='NOT TESTED',native_execution='BLOCKED_NO_COMPLETE_ENGINE_LEGAL_SET',
        limitation='Short correctness updates only;not a selected BC policy orPPO initialization artifact.')
    started=time.perf_counter()
    try:
        for step in range(args.updates):
            inputs,targets=tensors(train.batch(rng.integers(0,train.count,size=batch_size)),args.device)
            actor.train();outputs=actor(inputs,selected=targets['action'].clamp_min(0),teacher=targets['sequence'])
            loss=hierarchical_loss(outputs,targets)
            if not torch.isfinite(loss['total']):raise FloatingPointError('nonfinite hierarchical loss')
            optimizer.zero_grad(set_to_none=True);loss['total'].backward()
            norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),.5,error_if_nonfinite=True)
            optimizer.step();report['updates']+=1
            report['history'].append(dict(step=step+1,gradient_norm=float(norm),**{k:float(v.detach()) for k,v in loss.items()}))
        actor.eval();data=validation.batch(np.arange(min(batch_size,validation.count)));inputs,targets=tensors(data,args.device)
        with torch.no_grad():
            outputs=actor(inputs,selected=targets['action'].clamp_min(0),teacher=targets['sequence'])
            report['validation_smoke']={k:float(v) for k,v in hierarchical_loss(outputs,targets).items()}
        save_hierarchical(actor,out/'checkpoint.json',source,train.manifest['catalog'],
            dict(config=cfg,data_sha256=sha256(args.data),updates=args.updates,optimizer_saved=False))
        restored,_=load_hierarchical(out/'checkpoint.json',args.device)
        with torch.no_grad():
            again=restored(inputs,selected=targets['action'].clamp_min(0),teacher=targets['sequence'])
            for key in outputs:torch.testing.assert_close(outputs[key],again[key],rtol=0,atol=0)
        report.update(status='PASS_SUPERVISED_ENGINEERING_ONLY',save_load='PASS',seconds=time.perf_counter()-started)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:save_json(out/'result.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

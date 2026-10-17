"""One explicitly bounded semantic BC run; no search or automatic continuation."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from soku_rl.rl.factorized_bc import load_factorized, save_factorized, factorized_loss, rare_weights
from soku_rl.rl.hierarchical_data import SemanticDataset, tensors
from soku_rl.rl.hierarchical_training import materialize, evaluate
from soku_rl.rl.semantic_sampling import stratified_order, RATIOS
from soku_rl.replay.human_dataset import sha256, save_json


def validate_config(cfg):
    if (cfg['format'] != 'factorized_training_v2' or not 1 <= cfg['cycles'] <= 10
            or not 1 <= cfg['batch_size'] <= 128 or not 1 <= cfg['max_updates'] <= 5000
            or cfg['learning_rate'] != .00003 or cfg['termination_weight'] != .3
            or cfg['future_weight'] != .15 or cfg['rare_weight_max'] != 4.
            or cfg['sampling_ratios'] != RATIOS.tolist() or cfg['optimizer'] != 'fresh_adam'
            or cfg['selection'] != 'full_validation_weighted_objective'
            or not 1 <= cfg['draws_per_cycle'] <= 41075
            or (cfg['row_cap'], cfg['replay_cap'], cfg['chain_cap']) != (2, 256, 32)):
        raise ValueError('explicit bounded approved factorized training contract required')
    if math.ceil(cfg['draws_per_cycle']/cfg['batch_size'])*cfg['cycles'] > cfg['max_updates']:
        raise ValueError('declared complete cycles exceed optimizer budget')


def run_training(config, output, device='cuda'):
    root = Path(__file__).resolve().parents[4]
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    validate_config(cfg)
    out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(root/'rl/runs'):
        raise ValueError('new project-local output required')
    out.mkdir(parents=True)
    for folder in ('temp', 'cuda-cache', 'torch-cache', 'checkpoints'):
        (out/folder).mkdir()
    os.environ.update(TEMP=str(out/'temp'), TMP=str(out/'temp'), CUDA_CACHE_PATH=str(out/'cuda-cache'),
                      TORCH_HOME=str(out/'torch-cache'))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    started = time.perf_counter()
    def git(*args):
        proc = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    report = dict(status='RUNNING', config=cfg, updates=0, cycles=[], device=str(device),
        initial_sha256=sha256(cfg['initial_checkpoint']), data_sha256=sha256(cfg['data']),
        rl_commit=git('rev-parse','HEAD'), rl_dirty=git('status','--short','--','rl'),
        platform_reference_commit=git('rev-parse','origin/platform'),
        platform_source_identity='source-manifest.json;not assumed clean from remote ref',
        sealed_test_used=False, strength='NOT TESTED', optimizer='FRESH_ADAM',
        selection='Full natural validation, train-only weighted 1/.3/.15 objective; includes initial checkpoint')
    # Hash actual module sources; ignored local formatting changes are explicit.
    identity = {str(p.relative_to(root)):sha256(p) for base in ('rl/src','rl/tools','rl/config','platform/src')
        for p in (root/base).rglob('*') if p.is_file() and p.suffix in ('.py','.ps1','.yaml','.yml')
        and '__pycache__' not in p.parts and 'local' not in p.relative_to(root).parts}
    save_json(out/'source-manifest.json', identity)
    report['source_manifest_sha256'] = sha256(out/'source-manifest.json')
    save_json(out/'result.json', report)
    try:
        training = SemanticDataset(cfg['data'], 'train')
        validation_set = SemanticDataset(cfg['data'], 'validation')
        train, validation = materialize(training), materialize(validation_set)
        report['resident_dataset_bytes'] = sum(a.nbytes for d in (train,validation) for a in d.values())
        actor, metadata = load_factorized(cfg['initial_checkpoint'], device)
        if metadata['catalog'] != training.manifest['catalog']:
            raise ValueError('model/data catalog mismatch')
        catalog = metadata['catalog']
        weights = rare_weights(torch.as_tensor(train['target_action']),
                               torch.as_tensor(train['target_action_mask']), actor.actions).to(device)
        report['class_weights'] = weights.cpu().tolist()
        optimizer = torch.optim.Adam(actor.parameters(), lr=cfg['learning_rate'])
        def assess(data):
            return evaluate(actor, data, catalog, cfg['batch_size'], device,
                            task_weights=(.3,.15), ce_weights=weights)
        def checkpoint(name, cycle):
            path = out/'checkpoints'/f'{name}.json'
            save_factorized(actor, path, metadata['source'], dict(config=cfg,cycle=cycle,
                updates=report['updates'],data_sha256=report['data_sha256'],rl_commit=report['rl_commit']))
            torch.save(optimizer.state_dict(), path.with_suffix('.adam.pt'))
            return dict(path=str(path),sha256=sha256(path),optimizer_sha256=sha256(path.with_suffix('.adam.pt')))
        initial = checkpoint('initial',0)
        report['initial_validation'] = assess(validation)
        report['best'] = dict(cycle=0,**initial,validation=report['initial_validation'])
        save_json(out/'result.json',report)
        with (out/'updates.jsonl').open('x',encoding='utf-8') as log:
            for cycle in range(1,cfg['cycles']+1):
                tick = time.perf_counter()
                order, roles, stats = stratified_order(train['supervision_strata'],train['replay_id'],train['chain_id'],
                    rng,cfg['draws_per_cycle'],row_cap=cfg['row_cap'],replay_cap=cfg['replay_cap'],chain_cap=cfg['chain_cap'])
                if stats['status'] != 'PASS':
                    raise RuntimeError('BLOCKED_CAPACITY: declared schedule cannot fit caps')
                np.savez_compressed(out/f'sampling-{cycle:02d}.npz',indices=order,roles=roles)
                for start in range(0,len(order),cfg['batch_size']):
                    if report['updates'] >= cfg['max_updates']:
                        raise RuntimeError('unexpected optimizer budget exhaustion')
                    indices = order[start:start+cfg['batch_size']]
                    x,y = tensors({k:v[indices] for k,v in train.items()},device)
                    actor.train()
                    losses = factorized_loss(actor(x,selected=y['action'].clamp_min(0),teacher=y['sequence']),y,weights)
                    optimizer.zero_grad(set_to_none=True)
                    losses['total'].backward()
                    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(),.5,error_if_nonfinite=True)
                    optimizer.step()
                    report['updates'] += 1
                    log.write(json.dumps(dict(cycle=cycle,update=report['updates'],gradient_norm=float(norm),
                        **{k:float(v.detach()) for k,v in losses.items()}))+'\n')
                log.flush()
                metrics = assess(validation)
                saved = checkpoint(f'cycle-{cycle:02d}',cycle)
                item = dict(cycle=cycle,updates=report['updates'],draws=len(order),sampling=stats,
                            seconds=time.perf_counter()-tick,validation=metrics,checkpoint=saved)
                report['cycles'].append(item)
                if metrics['selection_loss'] < report['best']['validation']['selection_loss']:
                    report['best'] = dict(cycle=cycle,**saved,validation=metrics)
                save_json(out/'result.json',report)
                save_json(out/'best.json',report['best'])
                print(json.dumps(item),flush=True)
        report['final_checkpoint'] = checkpoint('final',len(report['cycles']))
        actor.eval()
        restored,_ = load_factorized(report['final_checkpoint']['path'],device)
        x,_ = tensors({k:v[:8] for k,v in validation.items()},device)
        with torch.inference_mode():
            a,b = actor(x),restored(x)
            for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        report['final_reload'] = 'PASS_EXACT'
        best,_ = load_factorized(report['best']['path'],device)
        actor = best
        report['best_training_metrics'] = assess(train)
        report.update(status='PASS_BOUNDED_FACTORIZED_TRAINING',seconds=time.perf_counter()-started)
        if str(device).startswith('cuda'):
            report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    except BaseException as error:
        report.update(status='BLOCKED' if 'BLOCKED' in str(error) else 'FAIL',error=repr(error),
                      seconds=time.perf_counter()-started)
        raise
    finally:
        save_json(out/'result.json',report)
    print(json.dumps(dict(status=report['status'],best=report['best'],seconds=report['seconds'])),flush=True)

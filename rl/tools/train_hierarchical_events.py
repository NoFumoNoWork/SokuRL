"""Bounded event-BC continuation, full validation, immutable epoch models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
from omegaconf import OmegaConf
import torch

from soku_rl.replay.human_dataset import sha256, save_json
from soku_rl.rl.hierarchical_bc import load_hierarchical, save_hierarchical, hierarchical_loss
from soku_rl.rl.hierarchical_data import EventDataset, tensors
from soku_rl.rl.hierarchical_training import materialize, evaluate, validate_budget


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=Path('rl/config/hierarchical_training.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    validate_budget(cfg)
    if cfg['selection'] != 'full_validation_composite_loss':
        raise ValueError('explicit selection rule required')
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(root/'rl/runs'):
        raise ValueError('preserve existing runs; new project-local output required')
    out.mkdir(parents=True)
    for name in ('temp', 'cuda-cache', 'torch-cache'):
        (out/name).mkdir()
    os.environ.update(TEMP=str(out/'temp'), TMP=str(out/'temp'),
                      CUDA_CACHE_PATH=str(out/'cuda-cache'), TORCH_HOME=str(out/'torch-cache'))
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    started = time.perf_counter()
    report = dict(status='RUNNING', config=cfg, updates=0, event_visits=0, history=[],
                  optimizer='fresh Adam; previous smoke optimizer not saved',
                  native_execution='NOT TESTED FOR THIS CHECKPOINT', policy_strength='NOT TESTED',
                  sequence_metrics='teacher-forced conditional future NLL, not open-loop execution',
                  test_set='SEALED_NOT_USED', reward_and_ppo='UNCHANGED',
                  torch_version=torch.__version__, device=args.device)
    try:
        # Dirty source is part of the identity, not represented by HEAD alone.
        report['git'] = {name: subprocess.check_output(['git', *command], cwd=root, text=True)
                         for name, command in dict(head=['rev-parse', 'HEAD'],
                                                   branch=['branch', '--show-current'],
                                                   status=['status', '--porcelain', '--untracked-files=no']).items()}
        source_hashes = {}
        with zipfile.ZipFile(out/'source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for folder in ('rl/src', 'rl/tools', 'rl/config', 'rl/tests'):
                for path in sorted((root/folder).rglob('*')):
                    if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.ps1') and '__pycache__' not in path.parts:
                        relative = path.relative_to(root).as_posix()
                        archive.write(path, relative)
                        source_hashes[relative] = sha256(path)
        report['source_hashes'] = source_hashes
        report['source_archive_sha256'] = sha256(out/'source.zip')
        from game_runtime.identity import fingerprints
        report['platform_identity'] = fingerprints()
        report['initial_checkpoint_sha256'] = sha256(cfg['initial_checkpoint'])
        report['data_sha256'] = sha256(cfg['data'])
        train_set = EventDataset(cfg['data'], 'train')
        valid_set = EventDataset(cfg['data'], 'validation')
        catalog = train_set.manifest['catalog']
        actor, metadata = load_hierarchical(cfg['initial_checkpoint'], args.device)
        if metadata['catalog'] != catalog or valid_set.manifest['catalog'] != catalog:
            raise ValueError('checkpoint/data command contract differs')
        train = materialize(train_set)
        valid = materialize(valid_set)
        report['data_bytes'] = sum(v.nbytes for data in (train, valid) for v in data.values())
        report['train_rows'] = train_set.count
        report['validation_rows'] = valid_set.count
        report['initial_validation'] = evaluate(actor, valid, catalog, cfg['batch_size'], args.device)
        save_hierarchical(actor, out/'initial.json', metadata['source'], catalog,
                          dict(parent_sha256=report['initial_checkpoint_sha256'], updates=0))
        best = report['initial_validation']['selection_loss']
        report.update(best_checkpoint='initial.json', best_epoch=0)
        optimizer = torch.optim.Adam(actor.parameters(), lr=cfg['learning_rate'])
        save_json(out/'result.json', report)
        print(json.dumps(dict(phase='initial_validation', metrics=report['initial_validation'])), flush=True)
        for epoch in range(1, cfg['epochs']+1):
            epoch_start = time.perf_counter()
            order = rng.permutation(train_set.count)
            sums = {}; gradients = []
            actor.train()
            for start in range(0, len(order), cfg['batch_size']):
                indices = order[start:start+cfg['batch_size']]
                x, y = tensors({k: v[indices] for k, v in train.items()}, args.device)
                outputs = actor(x, selected=y['action'].clamp_min(0), teacher=y['sequence'])
                losses = hierarchical_loss(outputs, y)
                if not torch.isfinite(losses['total']):
                    raise FloatingPointError('nonfinite supervised loss')
                optimizer.zero_grad(set_to_none=True)
                losses['total'].backward()
                norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), .5, error_if_nonfinite=True)
                optimizer.step()
                gradients.append(float(norm))
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.) + float(value.detach())*len(indices)
                report['updates'] += 1
                report['event_visits'] += len(indices)
            metrics = evaluate(actor, valid, catalog, cfg['batch_size'], args.device)
            path = out/f'epoch-{epoch:04d}.json'
            save_hierarchical(actor, path, metadata['source'], catalog,
                              dict(config=cfg, epoch=epoch, updates=report['updates'],
                                   parent_sha256=report['initial_checkpoint_sha256'],
                                   data_sha256=report['data_sha256'], validation=metrics))
            # One immutable state per epoch enables audited interrupted-run recovery.
            torch.save(dict(optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                            model_sha256=sha256(path), epoch=epoch, updates=report['updates']),
                       out/f'epoch-{epoch:04d}-training.pt')
            entry = dict(epoch=epoch, updates=report['updates'], validation=metrics,
                         minibatch_loss_row_weighted={k: v/train_set.count for k, v in sums.items()},
                         gradient_norm_mean=float(np.mean(gradients)), gradient_norm_max=max(gradients),
                         seconds=time.perf_counter()-epoch_start)
            report['history'].append(entry)
            if metrics['selection_loss'] < best:
                best = metrics['selection_loss']
                report.update(best_checkpoint=path.name, best_epoch=epoch)
            report['seconds'] = time.perf_counter()-started
            save_json(out/'result.json', report)
            print(json.dumps(entry), flush=True)
        best_path = out/report['best_checkpoint']
        restored, selected = load_hierarchical(best_path, args.device)
        selected_metrics = evaluate(restored, valid, catalog, cfg['batch_size'], args.device)
        if abs(selected_metrics['selection_loss']-best) > 1e-6:
            raise AssertionError('selected model validation changed after reload')
        save_json(out/'best.json', selected)
        report.update(status='PASS_BOUNDED_SUPERVISED_TRAINING', best_validation=selected_metrics,
                      best_checkpoint_sha256=sha256(best_path), save_load='PASS',
                      seconds=time.perf_counter()-started,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0)
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), seconds=time.perf_counter()-started)
        raise
    finally:
        save_json(out/'result.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'updates', 'event_visits', 'best_epoch', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()

"""Bounded architecture acceptance, NOT formal training or model selection."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from soku_rl.rl.factorized_bc import (FactorizedActor, CATEGORIES, category_ids, factorized_loss,
    rare_weights, warm_start, save_factorized, load_factorized)
from soku_rl.rl.hierarchical_bc import load_hierarchical
from soku_rl.rl.hierarchical_data import SemanticDataset, tensors
from soku_rl.rl.hierarchical_training import materialize
from soku_rl.rl.semantic_sampling import stratified_order, RATIOS, STRATA
from soku_rl.replay.human_dataset import sha256, save_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('rl/config/hierarchical_factorized.yaml'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if (out.exists() or not out.is_relative_to(root / 'rl/runs') or cfg['formal_training']
            or cfg['format'] != 'hierarchical_bc_v2_factorized'
            or not 0 <= cfg['max_correctness_updates'] <= 4
            or [cfg[k] for k in ('choice_weight', 'termination_weight', 'future_weight', 'rare_weight_max')]
            != [1., .3, .15, 4.] or cfg['sampling']['ratios'] != RATIOS.tolist()
            or cfg['sampling']['strata'] != list(STRATA)
            or [cfg[k] for k in ('event_history', 'event_layers', 'event_width', 'event_heads',
                                  'input_history', 'sequence_horizon')] != [16, 3, 128, 4, 60, 3]):
        raise ValueError('new local output and approved bounded correctness contract required')
    out.mkdir(parents=True)
    for name in ('temp', 'cuda-cache', 'torch-cache'):
        (out / name).mkdir()
    os.environ.update(TEMP=str(out/'temp'), TMP=str(out/'temp'), CUDA_CACHE_PATH=str(out/'cuda-cache'),
                      TORCH_HOME=str(out/'torch-cache'))
    torch.set_num_threads(4)
    # Compare float32 CPU/CUDA semantics, not CUDA's optional TF32 convolution.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    report = dict(status='RUNNING', config=cfg, data_sha256=sha256(args.data),
        source_sha256=sha256(cfg['source_checkpoint']), warm_start_sha256=sha256(cfg['warm_start']),
        formal_training='NOT RUN', strength='NOT TESTED', sealed_test_used=False, steps=[],
        precision=dict(dtype='float32', tf32=False, mha_fastpath=False))
    save_json(out/'result.json', report)
    started = time.perf_counter()
    try:
        train_set = SemanticDataset(args.data, 'train')
        validation = SemanticDataset(args.data, 'validation')
        train = materialize(train_set)
        report['host_array_bytes'] = sum(a.nbytes for a in train.values())
        sam = cfg['sampling']
        order, roles, stats = stratified_order(train['supervision_strata'], train['replay_id'], train['chain_id'],
            rng, sam['diagnostic_draws'], row_cap=sam['row_cap'], replay_cap=sam['replay_cap'], chain_cap=sam['chain_cap'])
        report['sampler'] = stats
        np.savez_compressed(out/'sampling.npz', indices=order, roles=roles)
        if stats['status'] != 'PASS':
            raise RuntimeError('BLOCKED: requested draws cannot fit declared caps')
        catalog = train_set.manifest['catalog']
        weights = rare_weights(torch.as_tensor(train['target_action']), torch.as_tensor(train['target_action_mask']), len(catalog))
        report['weights'] = weights.tolist()
        report['category_counts'] = {name: int(np.sum(
            np.asarray(category_ids(catalog))[train['target_action'][train['target_action_mask']]] == i))
            for i, name in enumerate(CATEGORIES)}
        from stable_baselines3 import PPO
        original = PPO.load(cfg['source_checkpoint'], device='cpu')
        actor = FactorizedActor(original.policy.features_extractor, catalog)
        previous, old = load_hierarchical(cfg['warm_start'])
        if old['source_sha256'] != report['source_sha256'] or old['catalog'] != catalog:
            raise ValueError('warm-start source/catalog mismatch')
        report['warm_start'] = warm_start(actor, previous)
        del original, previous
        optimizer = torch.optim.Adam(actor.parameters(), lr=cfg['learning_rate'])
        save_factorized(actor, out/'initial.json', cfg['source_checkpoint'], dict(config=cfg, updates=0))
        for i in range(cfg['max_correctness_updates']):
            indices = order[i*cfg['batch_size']:(i+1)*cfg['batch_size']]
            if len(indices) != cfg['batch_size']:
                raise ValueError('insufficient bounded sample draws')
            inputs, targets = tensors({k: a[indices] for k, a in train.items()}, 'cpu')
            actor.train()
            outputs = actor(inputs, selected=targets['action'].clamp_min(0), teacher=targets['sequence'])
            loss = factorized_loss(outputs, targets, weights)
            optimizer.zero_grad(set_to_none=True)
            loss['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            report['steps'].append(dict(update=i+1, gradient_norm=float(norm),
                valid_labels={key: int(targets[key].sum()) for key in ('action_mask', 'termination_mask', 'sequence_mask')},
                **{k: float(v.detach()) for k, v in loss.items()}))
        actor.eval()
        inputs, targets = tensors(validation.batch(np.arange(min(16, validation.count))), 'cpu')
        with torch.inference_mode():
            before = actor(inputs)
            report['validation_forward'] = {k: float(v) for k, v in factorized_loss(before, targets, weights).items()}
        save_factorized(actor, out/'checkpoint.json', cfg['source_checkpoint'],
            dict(config=cfg, data_sha256=sha256(args.data), updates=len(report['steps']), purpose='CORRECTNESS_NOT_SELECTED_POLICY'))
        restored, _ = load_factorized(out/'checkpoint.json')
        with torch.inference_mode():
            after = restored(inputs)
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        report['save_reload'] = 'PASS_EXACT'
        if torch.cuda.is_available():
            restored.cuda()
            with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                gpu = restored({k: v.cuda() for k, v in inputs.items()})
            for key in before:
                torch.testing.assert_close(before[key], gpu[key].cpu(), rtol=1e-4, atol=2e-5)
            report['cuda_forward'] = 'PASS_TOLERANCE_1e-4_2e-5'
        else:
            report['cuda_forward'] = 'NOT TESTED_NO_CUDA'
        report.update(status='PASS_ARCHITECTURE_CORRECTNESS_ONLY', seconds=time.perf_counter()-started)
    except BaseException as error:
        report.update(status='BLOCKED' if 'BLOCKED' in str(error) else 'FAIL', error=repr(error))
        raise
    finally:
        save_json(out/'result.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

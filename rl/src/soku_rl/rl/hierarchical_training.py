"""Full-partition metrics for bounded event BC; no test-set access."""
import numpy as np
import torch
import torch.nn.functional as F

from soku_rl.rl.hierarchical_data import tensors


def materialize(dataset):
    """Read each verified shard once, preserving all rows and masks."""
    pieces = []
    start = 0
    for end in dataset.ends:
        pieces.append(dataset.batch(np.arange(start, end)))
        start = int(end)
    dataset.cache.clear()
    return {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}


def epoch_order(data, catalog, rng, mode='uniform'):
    """Resample known choices only; preserve every non-choice row and all masks."""
    if mode not in ('uniform', 'balanced_attack_movement'):
        raise ValueError('unknown event sampling mode')
    labels = np.asarray(data['target_action'])
    known = np.asarray(data['target_action_mask'], bool)
    if np.any(labels[known] < 0) or np.any(labels[known] >= len(catalog)):
        raise ValueError('supervised action outside command catalog')
    is_attack = np.zeros(len(labels), bool)
    attacks = np.array([not c['continuous'] for c in catalog], bool)
    is_attack[known] = attacks[labels[known]]
    groups = dict(attack=np.flatnonzero(known & is_attack),
                  movement=np.flatnonzero(known & ~is_attack), context=np.flatnonzero(~known))
    if mode == 'uniform':
        order = rng.permutation(len(labels))
    else:
        if not len(groups['attack']) or not len(groups['movement']):
            raise ValueError('balanced sampling requires confirmed attack and movement choices')
        count = int(known.sum())
        quotas = dict(attack=(count+1)//2, movement=count//2, context=len(groups['context']))
        pieces = [rng.choice(groups[k], size=n, replace=n>len(groups[k]))
                  if k != 'context' else groups[k].copy() for k,n in quotas.items()]
        order = rng.permutation(np.concatenate(pieces))
    selected_known = known[order]
    selected_attack = is_attack[order]
    stats = dict(mode=mode, rows=len(order), draws={}, unique={})
    for k, selected in [('attack', selected_known & selected_attack),
                        ('movement', selected_known & ~selected_attack), ('context', ~selected_known)]:
        stats['draws'][k] = int(selected.sum())
        stats['unique'][k] = int(len(np.unique(order[selected])))
    return order, stats


def evaluate(actor, data, catalog, batch_size, device, *, task_weights=(.5,.5), ce_weights=None):
    actor.eval()
    totals = {}
    counts = {}
    correct = {}
    stop_confusion = dict(tp=0, fp=0, tn=0, fn=0)
    attack_confusion = dict(tp=0, fp=0, tn=0, fn=0)
    predicted_counts = np.zeros(len(catalog), np.int64)
    label_counts = np.zeros(len(catalog), np.int64)
    attack = torch.tensor([not c['continuous'] for c in catalog], device=device)

    weighted_totals = {}

    def accumulate(name, logits, labels, mask, binary=False, predictions=None):
        mask = mask.bool()
        n = int(mask.sum())
        if not n:
            return
        values, targets = logits[mask], labels[mask]
        loss = (F.binary_cross_entropy_with_logits(values, targets.float(), reduction='sum')
                if binary else F.cross_entropy(values, targets.long(), reduction='sum'))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite validation {name}')
        predictions = (predictions[mask] if predictions is not None else
                       values >= 0 if binary else values.argmax(-1))
        if ce_weights is not None and not binary and values.shape[-1] == len(ce_weights):
            weighted = F.cross_entropy(values, targets.long(), weight=ce_weights, reduction='sum')
            weighted_totals[name] = weighted_totals.get(name, 0.) + float(weighted)
        totals[name] = totals.get(name, 0.) + float(loss)
        counts[name] = counts.get(name, 0) + n
        correct[name] = correct.get(name, 0) + int((predictions == targets).sum())

    with torch.no_grad():
        for start in range(0, len(data['state']), batch_size):
            x, y = tensors({k: v[start:start+batch_size] for k, v in data.items()}, device)
            outputs = actor(x, selected=y['action'].clamp_min(0), teacher=y['sequence'])
            mask = y['action_mask'].bool()
            is_attack = attack[y['action'].clamp_min(0).long()]
            chosen = outputs.get('choice_id', outputs['action'].argmax(-1))
            predicted = chosen[mask]
            predicted_counts += np.bincount(predicted.cpu().numpy(), minlength=len(catalog))
            label_counts += np.bincount(y['action'][mask].long().cpu().numpy(), minlength=len(catalog))
            predicted_attack, actual_attack = attack[predicted], is_attack[mask]
            for key, subset in [('tp', predicted_attack & actual_attack), ('fp', predicted_attack & ~actual_attack),
                                ('tn', ~predicted_attack & ~actual_attack), ('fn', ~predicted_attack & actual_attack)]:
                attack_confusion[key] += int(subset.sum())
            for name, subset in [('action', mask), ('attack', mask & is_attack), ('movement', mask & ~is_attack)]:
                accumulate(name, outputs['action'], y['action'], subset, predictions=chosen)
            if 'category' in outputs:
                categories = actor.choice.mapping[y['action'].clamp_min(0).long()]
                accumulate('category', outputs['category'], categories, mask)
            accumulate('termination', outputs['termination'], y['termination'], y['termination_mask'], True)
            tm = y['termination_mask'].bool()
            actual = y['termination'][tm].bool()
            predicted = outputs['termination'][tm] >= 0
            for key, subset in [('tp', predicted & actual), ('fp', predicted & ~actual),
                                ('tn', ~predicted & ~actual), ('fn', ~predicted & actual)]:
                stop_confusion[key] += int(subset.sum())
            accumulate('sequence_teacher_forced', outputs['sequence'], y['sequence'], y['sequence_mask'],
                       predictions=outputs.get('sequence_choice_ids'))
            free = actor(x)
            accumulate('sequence_free_running', free['sequence'], y['sequence'], y['sequence_mask'],
                       predictions=free.get('sequence_choice_ids'))
            for i in range(3):
                accumulate(f'future_{i+1}_teacher_forced', outputs['sequence'][:, i],
                           y['sequence'][:, i], y['sequence_mask'][:, i],
                           predictions=(outputs['sequence_choice_ids'][:,i] if 'sequence_choice_ids' in outputs else None))
    metrics = {k: dict(count=counts[k], nll=totals[k]/counts[k], accuracy=correct[k]/counts[k]) for k in counts}
    for key,value in weighted_totals.items():metrics[key]['weighted_nll']=value/counts[key]
    required = ('action', 'termination', 'sequence_teacher_forced')
    if any(k not in metrics for k in required):
        raise ValueError('full validation requires all three supervised tasks')
    field='weighted_nll' if ce_weights is not None else 'nll'
    metrics['selection_loss'] = (metrics['action'][field] + task_weights[0]*metrics['termination']['nll']
                                 + task_weights[1]*metrics['sequence_teacher_forced'][field])
    metrics['rows'] = len(data['state'])
    metrics['stop_confusion'] = stop_confusion
    n = sum(stop_confusion.values())
    metrics['always_continue_accuracy'] = (stop_confusion['tn']+stop_confusion['fp'])/n
    metrics['attack_confusion'] = attack_confusion
    metrics['predicted_commands'] = {c.get('name',str(i)):int(n) for i,(c,n) in enumerate(zip(catalog,predicted_counts)) if n}
    metrics['label_commands'] = {c.get('name',str(i)):int(n) for i,(c,n) in enumerate(zip(catalog,label_counts)) if n}
    metrics['constant_majority_accuracy'] = float(label_counts.max()/label_counts.sum())
    if 'attack' in metrics and 'movement' in metrics:
        metrics['balanced_action_accuracy'] = .5*(metrics['attack']['accuracy']+metrics['movement']['accuracy'])
    return metrics


def validate_budget(config):
    if config['format'] != 'hierarchical_training_v1':
        raise ValueError('explicit supervised training profile required')
    if not 1 <= config['epochs'] <= 10 or not 1 <= config['batch_size'] <= 128:
        raise ValueError('bounded budget: 1..10 epochs, batch <=128')
    if config['learning_rate'] != .00003:
        raise ValueError('this continuation preserves learning rate')
    if config['termination_weight'] != .5 or config['sequence_weight'] != .5:
        raise ValueError('approved objective must be preserved')
    if not 1 <= config['max_updates'] <= 5000:
        raise ValueError('at most 5000 optimizer updates')
    if config.get('sampling','uniform') not in ('uniform', 'balanced_attack_movement'):
        raise ValueError('explicit supported sampler required')
    resuming = bool(config.get('resume_training_report'))
    if resuming != (config.get('optimizer') == 'resume_adam_from_bound_selected_checkpoint'):
        raise ValueError('resume profile and bound training report must be specified together')


def restore_optimizer(optimizer, checkpoint, report_path, learning_rate):
    """Restore only the hash-bound Adam sidecar of a completed selected HBC model."""
    import json
    from pathlib import Path
    from soku_rl.replay.human_dataset import sha256
    checkpoint = Path(checkpoint).resolve(strict=True)
    report_path = Path(report_path).resolve(strict=True)
    previous = json.loads(report_path.read_text())
    if previous.get('status') != 'PASS_BOUNDED_SUPERVISED_TRAINING':
        raise ValueError('optimizer requires a completed supervised training report')
    best = previous['best']
    if Path(best['path']).resolve() != checkpoint or sha256(checkpoint) != best['sha256']:
        raise ValueError('optimizer report does not bind the initial selected checkpoint')
    sidecar = checkpoint.with_suffix('.adam.pt')
    if sha256(sidecar) != best['optimizer_sha256']:
        raise ValueError('optimizer sidecar changed')
    state = torch.load(sidecar, map_location='cpu', weights_only=True)
    if any(group['lr'] != learning_rate for group in state['param_groups']):
        raise ValueError('restored learning rate differs; no silent override')
    optimizer.load_state_dict(state)
    steps = [float(item['step']) for item in optimizer.state.values() if 'step' in item]
    return dict(status='PASS', report=str(report_path), report_sha256=sha256(report_path),
        sidecar=str(sidecar), sidecar_sha256=sha256(sidecar), state_count=len(optimizer.state),
        minimum_step=min(steps) if steps else 0., maximum_step=max(steps) if steps else 0.)


def run_training(config_path, data_path, out, device, root):
    """Single bounded continuation; unchanged model and masked objective."""
    import json
    import os
    from pathlib import Path
    import subprocess
    import time
    import zipfile
    from omegaconf import OmegaConf
    from soku_rl.rl.hierarchical_bc import load_hierarchical, save_hierarchical, hierarchical_loss
    from soku_rl.rl.hierarchical_data import EventDataset
    from soku_rl.replay.human_dataset import sha256, save_json

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    validate_budget(cfg)
    if out.exists() or not out.is_relative_to(root/'rl/runs'):
        raise ValueError('preserve outputs; use a NEW project-local run')
    if Path(data_path).resolve() != Path(cfg['data']).resolve():
        raise ValueError('data differs from explicit profile')
    out.mkdir(parents=True)
    for name in ('temp', 'cuda-cache', 'torch-cache', 'checkpoints'):
        (out/name).mkdir()
    os.environ.update(TEMP=str(out/'temp'), TMP=str(out/'temp'),
                      CUDA_CACHE_PATH=str(out/'cuda-cache'), TORCH_HOME=str(out/'torch-cache'))
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    started = time.perf_counter()
    report = dict(status='RUNNING', config=cfg, data_sha256=sha256(data_path), updates=0,
                  epochs=[], optimizer='fresh Adam; previous optimizer moments unavailable',
                  initialization=str(Path(cfg['initial_checkpoint']).resolve()),
                  initialization_sha256=sha256(cfg['initial_checkpoint']),
                  strength='NOT TESTED', sealed_test_used=False)
    def git(*args):
        p = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True)
        return dict(returncode=p.returncode, output=p.stdout.strip(), error=p.stderr.strip())
    report['git'] = {name: git(*args) for name, args in {
        'head': ('rev-parse', 'HEAD'), 'branch': ('branch', '--show-current'),
        'rl_last_commit': ('log', '-1', '--format=%H', '--', 'rl'),
        'platform_last_commit': ('log', '-1', '--format=%H', '--', 'platform'),
        'dirty_tracked': ('status', '--short', '--untracked-files=no')}.items()}
    # Actual working sources are needed: this workspace includes ignored code.
    fingerprints = {}
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ('rl/src', 'rl/tools', 'rl/tests', 'rl/config'):
            for p in sorted((root/folder).rglob('*')):
                if p.is_file() and p.suffix in ('.py', '.ps1', '.yaml', '.yml'):
                    relative = p.relative_to(root).as_posix()
                    archive.write(p, relative)
                    fingerprints[relative] = sha256(p)
    save_json(out/'source_manifest.json', fingerprints)
    report['source_archive_sha256'] = sha256(out/'source.zip')
    save_json(out/'result.json', report)
    try:
        train_ds = EventDataset(data_path, 'train')
        val_ds = EventDataset(data_path, 'validation')
        train = materialize(train_ds)
        validation = materialize(val_ds)
        report['resident_dataset_bytes'] = sum(v.nbytes for d in (train, validation) for v in d.values())
        actor, metadata = load_hierarchical(cfg['initial_checkpoint'], device)
        if metadata['catalog'] != train_ds.manifest['catalog']:
            raise ValueError('checkpoint and dataset command catalogs differ')
        catalog = metadata['catalog']
        source = metadata['source']
        optimizer = torch.optim.Adam(actor.parameters(), lr=cfg['learning_rate'])
        if cfg.get('resume_training_report'):
            report['optimizer_restore'] = restore_optimizer(optimizer, cfg['initial_checkpoint'],
                cfg['resume_training_report'], cfg['learning_rate'])
            report['optimizer'] = 'Restored hash-bound Adam moments; new dataset and sampling seed'
        initial_parameters = {k: v.detach().cpu().clone() for k,v in actor.named_parameters()}
        def checkpoint(name, epoch):
            path = out/'checkpoints'/f'{name}.json'
            save_hierarchical(actor, path, source, catalog, dict(config=cfg, epoch=epoch,
                updates=report['updates'], data_sha256=report['data_sha256'], optimizer_saved=True))
            torch.save(optimizer.state_dict(), path.with_suffix('.adam.pt'))
            return dict(path=str(path), sha256=sha256(path),
                        optimizer_sha256=sha256(path.with_suffix('.adam.pt')))
        report['initial_checkpoint'] = checkpoint('initial', 0)
        report['initial_validation'] = evaluate(actor, validation, catalog, cfg['batch_size'], device)
        report['best'] = dict(epoch=0, **report['initial_checkpoint'], validation=report['initial_validation'])
        save_json(out/'result.json', report)
        with (out/'updates.jsonl').open('x', encoding='utf-8') as log:
            for epoch in range(1, cfg['epochs']+1):
                tick = time.perf_counter()
                order, sampling = epoch_order(train, catalog, rng, cfg.get('sampling','uniform'))
                seen = 0
                for start in range(0, len(order), cfg['batch_size']):
                    if report['updates'] >= cfg['max_updates']:break
                    indices = order[start:start+cfg['batch_size']]
                    x,y = tensors({k:v[indices] for k,v in train.items()}, device)
                    actor.train()
                    output = actor(x, selected=y['action'].clamp_min(0), teacher=y['sequence'])
                    losses = hierarchical_loss(output, y)
                    if not torch.isfinite(losses['total']):raise FloatingPointError('nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True)
                    losses['total'].backward()
                    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), .5, error_if_nonfinite=True)
                    optimizer.step()
                    report['updates'] += 1
                    seen += len(indices)
                    log.write(json.dumps(dict(update=report['updates'], epoch=epoch,
                        gradient_norm=float(norm), **{k:float(v.detach()) for k,v in losses.items()}))+'\n')
                log.flush()
                metrics = evaluate(actor, validation, catalog, cfg['batch_size'], device)
                saved = checkpoint(f'epoch-{epoch:02d}', epoch)
                item = dict(epoch=epoch, updates=report['updates'], rows_seen=seen,
                            seconds=time.perf_counter()-tick, sampling=sampling,
                            validation=metrics, checkpoint=saved)
                report['epochs'].append(item)
                if metrics['selection_loss'] < report['best']['validation']['selection_loss']:
                    report['best'] = dict(epoch=epoch, **saved, validation=metrics)
                save_json(out/'result.json', report)
                save_json(out/'best.json', report['best'])
                print(json.dumps(item), flush=True)
                if report['updates'] >= cfg['max_updates']:break
        report['final_checkpoint'] = checkpoint('final', len(report['epochs']))
        report['parameter_delta_l2'] = float(sum((v.detach().cpu()-initial_parameters[k]).double().square().sum()
            for k,v in actor.named_parameters()).sqrt())
        restored,_ = load_hierarchical(report['final_checkpoint']['path'], device)
        actor.eval()
        x,y = tensors({k:v[:8] for k,v in validation.items()}, device)
        with torch.no_grad():
            before, after = actor(x), restored(x)
            for key in before:torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        report.update(status='PASS_BOUNDED_SUPERVISED_TRAINING', save_reload='PASS', seconds=time.perf_counter()-started)
        if str(device).startswith('cuda'):report['cuda_peak_bytes'] = torch.cuda.max_memory_allocated()
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), seconds=time.perf_counter()-started)
        raise
    finally:save_json(out/'result.json', report)
    print(json.dumps(dict(status=report['status'], best=report['best'], seconds=report['seconds'])), flush=True)

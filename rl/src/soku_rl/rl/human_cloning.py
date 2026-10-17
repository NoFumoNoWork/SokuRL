"""Actor-only human BC; no synthetic teacher, reward or return targets."""
from collections import OrderedDict
from pathlib import Path
import time

import numpy as np
import torch

from soku_rl.replay.human_dataset import save_json, sha256
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.demonstration_loss import weighted_action_loss, weighted_validation_nll
from soku_rl.rl.learner import create_learner, parameter_hash


class HumanDataset:
    """Sample accepted rows without ever reconstructing history after masking."""
    def __init__(self, manifest_path, acceptance_path, cache_bytes=128*1024**2):
        import json
        self.path = Path(manifest_path).resolve(strict=True)
        self.manifest = json.loads(self.path.read_text())
        acceptance = json.loads(Path(acceptance_path).read_text())
        if (self.manifest['schema'] != 'human_replay_dataset_v1'
                or self.manifest['status'] != 'PASS_EXPORT_ONLY'
                or self.manifest['partition'] not in ('train','validation')
                or acceptance['status'] != 'PASS_PREFIX_PILOT'
                or acceptance['manifest_sha256'] != sha256(self.path)):
            raise ValueError('human BC requires hash-bound prefix acceptance,not a rule-AI manifest')
        self.acceptance = acceptance
        self.cache, self.cache_bytes, self.cached_bytes = OrderedDict(), cache_bytes, 0
        self.shards, counts = [], []
        for case in self.manifest['replays']:
            if case['status'] != 'PASS_EXPORT' or sha256(case['source']) != case['source_sha256']:
                raise ValueError('human source changed')
            for shard in case['shards']:
                path = self.path.parent/str(case['replay_id'])/shard['path']
                if not path.resolve().is_relative_to(self.path.parent) or sha256(path) != shard['sha256']:
                    raise ValueError('human shard identity changed')
                self.shards.append(path)
                counts.append(shard['supervised'])
        self.ends = np.cumsum(counts)
        self.count = int(self.ends[-1]) if len(self.ends) else 0
        if not self.count:
            raise ValueError('no human supervision')

    def shard(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        with np.load(self.shards[index],allow_pickle=False) as data:
            mask = data['supervised']
            result = {key:data[key][mask] for key in (
                'observations','actions','changed','previous_action')}
        if len(result['actions']) != self.ends[index]-(self.ends[index-1] if index else 0):
            raise ValueError('accepted shard row count changed')
        if (not np.isfinite(result['observations']).all() or np.any(result['actions'] < 0)
                or np.any(result['actions'] >= 90)):
            raise ValueError('invalid accepted human samples')
        size = sum(v.nbytes for v in result.values())
        while self.cache and self.cached_bytes+size > self.cache_bytes:
            _, previous = self.cache.popitem(last=False)
            self.cached_bytes -= sum(v.nbytes for v in previous.values())
        if size <= self.cache_bytes:
            self.cache[index] = result
            self.cached_bytes += size
        return result

    def batch(self, indices):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= self.count)):
            raise ValueError('invalid human dataset indices')
        shards = np.searchsorted(self.ends,indices,side='right')
        result = None
        for shard in np.unique(shards):
            data = self.shard(int(shard))
            positions = np.flatnonzero(shards == shard)
            rows = indices[positions]-(self.ends[shard-1] if shard else 0)
            if result is None:
                result = {k:np.empty((len(indices),*v.shape[1:]),dtype=v.dtype) for k,v in data.items()}
            for key in result:
                result[key][positions] = data[key][rows]
        return result


class HumanCollection:
    """Reference accepted datasets without copying shards or faking acceptance."""
    def __init__(self,path):
        import json
        self.path = Path(path).resolve(strict=True)
        descriptor = json.loads(self.path.read_text())
        if descriptor.get('schema')!='human_replay_collection_v1' or not descriptor['datasets']:
            raise ValueError('explicit human collection required')
        self.shards,self.parts = [],[]
        replays,players,seen_ids,seen_hashes = [],set(),set(),set()
        checkpoint = partition = None
        for item in descriptor['datasets']:
            for key in ('manifest','acceptance'):
                if sha256(item[key])!=item[key+'_sha256']:
                    raise ValueError('collection constituent changed')
            part = HumanDataset(item['manifest'],item['acceptance'])
            identity = part.manifest['identity']['source_checkpoint_sha256']
            if partition is not None and (part.manifest['partition']!=partition or identity!=checkpoint):
                raise ValueError('collection mixes partition or checkpoint contracts')
            partition,checkpoint = part.manifest['partition'],identity
            for case in part.manifest['replays']:
                if case['replay_id'] in seen_ids or case['source_sha256'] in seen_hashes:
                    raise ValueError('collection duplicates a source replay')
                seen_ids.add(case['replay_id']);seen_hashes.add(case['source_sha256'])
                replays.append(case)
            self.parts.append(part);self.shards.extend(part.shards)
            players.update(part.acceptance['players'])
        self.manifest = dict(schema='human_replay_collection_v1',partition=partition,replays=replays,
                             identity=dict(source_checkpoint_sha256=checkpoint))
        self.acceptance = dict(players=sorted(players),source_acceptances=descriptor['datasets'])


def score_human(model, dataset, batch_size):
    totals = dict(nll=0.,correct=0,entropy=0.,changed_nll=0.,changed_correct=0,changed_samples=0,
                  previous_correct=0,previous_eligible=0)
    predicted, actual = np.zeros(90,dtype=np.int64), np.zeros(90,dtype=np.int64)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0,dataset.count,batch_size):
            data = dataset.batch(np.arange(start,min(start+batch_size,dataset.count)))
            x = torch.as_tensor(data['observations'],device=model.device)
            y = torch.as_tensor(data['actions'],device=model.device)
            changed = torch.as_tensor(data['changed'],device=model.device,dtype=torch.bool)
            distribution = model.policy.get_distribution(x)
            nll = -distribution.log_prob(y)
            mode = distribution.distribution.probs.argmax(dim=1)
            if not torch.isfinite(nll).all():
                raise FloatingPointError('nonfinite human validation likelihood')
            totals['nll'] += float(nll.sum())
            totals['correct'] += int((mode == y).sum())
            totals['entropy'] += float(distribution.entropy().sum())
            totals['changed_nll'] += float(nll[changed].sum())
            totals['changed_correct'] += int(((mode == y)&changed).sum())
            totals['changed_samples'] += int(changed.sum())
            previous = data['previous_action']
            totals['previous_eligible'] += int((previous >= 0).sum())
            totals['previous_correct'] += int(((previous >= 0)&(previous == data['actions'])).sum())
            predicted += np.bincount(mode.cpu().numpy(),minlength=90)
            actual += np.bincount(data['actions'],minlength=90)
    result = dict(samples=dataset.count,nll=totals['nll']/dataset.count,
        accuracy=totals['correct']/dataset.count,entropy=totals['entropy']/dataset.count,
        changed_samples=totals['changed_samples'],predicted_counts=predicted.tolist(),label_counts=actual.tolist(),
        previous_input_baseline=dict(eligible=totals['previous_eligible'],
            coverage=totals['previous_eligible']/dataset.count,
            accuracy=totals['previous_correct']/totals['previous_eligible'] if totals['previous_eligible'] else None))
    if totals['changed_samples']:
        result.update(changed_nll=totals['changed_nll']/totals['changed_samples'],
                      changed_accuracy=totals['changed_correct']/totals['changed_samples'])
    commands = np.array([a*64+b for a in range(9) for b in (0,1,2,4,8,16,32,9,10,12)])
    result['input_categories'] = {name:dict(label_fraction=float(actual[mask].sum()/dataset.count),
        predicted_fraction=float(predicted[mask].sum()/dataset.count)) for name,mask in (
            ('melee',(commands&1)!=0),('shooting',(commands&6)!=0),
            ('dash',(commands&8)!=0),('direction',commands//64!=4),('cards',(commands&48)!=0))}
    return result


def fit_human(interface, algorithm, source, train, validation, directory, device, seed,
              epochs=10,batch_size=128,change_weight=1.,patience=2):
    if (not 1 <= epochs <= 10 or train.count > 1000000 or source['kind'] != 'weights'
            or train.manifest['partition'] != 'train' or validation.manifest['partition'] != 'validation'):
        raise ValueError('human BC budget,initialization or partitions invalid')
    if train.manifest['identity']['source_checkpoint_sha256'] != sha256(source['path']):
        raise ValueError('human data and initialization checkpoint differ')
    directory = Path(directory)
    model, identity = create_learner(ObservationContractEnv(interface),interface,algorithm,source,device,seed)
    if model.policy.optimizer.state:
        raise ValueError('human BC must start with a fresh optimizer')
    from stable_baselines3 import PPO
    original = PPO.load(source['path'],device='cpu')
    initial_hash = parameter_hash(model.policy)
    if initial_hash != parameter_hash(original.policy):
        raise ValueError('BC initialization differs from exact source policy')
    torch.manual_seed(seed)
    initial_parameters = {k:v.detach().cpu().clone() for k,v in model.policy.named_parameters()}
    report = dict(schema='human_behavior_cloning_v1',status='RUNNING',success=False,
        source=identity,initial_parameter_sha256=initial_hash,optimizer_initial_states=0,
        value_coef=0.,returns_supervised=False,train_frames=train.count,validation_frames=validation.count,
        epochs_limit=epochs,batch_size=batch_size,action_change_weight=change_weight,patience=patience,
        device=str(model.device),seed=seed,updates=0,history=[],manifest_hashes={
            'train':sha256(train.path),'validation':sha256(validation.path)},critic_risk=
            'Value head receives no BC gradient;shared features may change,so PPO critic must be revalidated.')
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    try:
        model.save(directory/'initial.zip')
        baseline = {name:score_human(model,data,batch_size) for name,data in [('train',train),('validation',validation)]}
        majority = int(np.argmax(baseline['train']['label_counts']))
        report['constant_action_baseline'] = dict(action=majority,
            train_accuracy=baseline['train']['label_counts'][majority]/train.count,
            validation_accuracy=baseline['validation']['label_counts'][majority]/validation.count)
        best = weighted_validation_nll(baseline['validation'],validation.count,change_weight)
        report.update(best_epoch=0,best_validation_nll=best)
        report['history'].append(dict(epoch=0,**baseline))
        model.save(directory/'best.zip')
        stale = 0
        for epoch in range(1,epochs+1):
            model.policy.set_training_mode(True)
            order = rng.permutation(train.count)
            norms = []
            for start in range(0,train.count,batch_size):
                data = train.batch(order[start:start+batch_size])
                x = torch.as_tensor(data['observations'],device=model.device)
                y = torch.as_tensor(data['actions'],device=model.device)
                changed = torch.as_tensor(data['changed'],device=model.device,dtype=torch.bool)
                nll = -model.policy.get_distribution(x).log_prob(y)
                loss = weighted_action_loss(nll,changed,torch.ones_like(changed),change_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite actor-only BC loss')
                model.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.policy.parameters(),model.max_grad_norm,
                                                     error_if_nonfinite=True)
                model.policy.optimizer.step()
                norms.append(float(norm));report['updates'] += 1
            metrics = {name:score_human(model,data,batch_size) for name,data in [('train',train),('validation',validation)]}
            value = weighted_validation_nll(metrics['validation'],validation.count,change_weight)
            report['history'].append(dict(epoch=epoch,**metrics,gradient_norm_mean=float(np.mean(norms)),
                elapsed_seconds=time.perf_counter()-started))
            if value < best:
                best,stale = value,0
                report.update(best_epoch=epoch,best_validation_nll=best)
                model.save(directory/'best.zip')
            else:
                stale += 1
            save_json(directory/'progress.json',report)
            print(f'Human BC epoch={epoch} train_nll={metrics["train"]["nll"]:.4f} validation_nll={value:.4f}',flush=True)
            if stale >= patience:
                break
        model.save(directory/'final.zip')
        report.update(status='PASS_OFFLINE_TRAINING_NOT_POLICY_IMPROVEMENT',success=True,
            final_parameter_sha256=parameter_hash(model.policy),seconds=time.perf_counter()-started,
            parameter_delta_l2={k:float((v.detach().cpu()-initial_parameters[k]).norm())
                                for k,v in model.policy.named_parameters()},
            checkpoints={name:sha256(directory/f'{name}.zip') for name in ('initial','best','final')},
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if model.device.type=='cuda' else None)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),seconds=time.perf_counter()-started)
        raise
    finally:
        save_json(directory/'result.json',report)
    return report

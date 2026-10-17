"""Explicit BC context sidecar; the original PPO actor remains loadable."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from soku_rl.replay.human_dataset import sha256, save_json


FORMAT = 'human_multiscale_actor_v1'


class HistoryContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.recent = nn.Sequential(nn.Linear(5*185, 128), nn.Tanh())
        self.distant = nn.Sequential(nn.Linear(3*40, 64), nn.Tanh())
        # Flattening after convolution preserves all60 ordered positions.
        self.inputs = nn.Sequential(nn.Conv1d(8, 32, 5, padding=2), nn.Tanh(),
                                    nn.Conv1d(32, 32, 5, padding=2), nn.Tanh(),
                                    nn.Flatten(), nn.Linear(60*32, 64), nn.Tanh())
        self.residual = nn.Linear(256, 128)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, recent, distant, inputs):
        return self.residual(torch.cat((self.recent(recent.flatten(1)),
            self.distant(distant.flatten(1)), self.inputs(inputs.transpose(1, 2))), dim=-1))


class SequenceActor(nn.Module):
    def __init__(self, policy, variant):
        super().__init__()
        if variant not in ('A', 'B', 'C', 'D') or not policy.share_features_extractor:
            raise ValueError('A/B/C require the original shared-feature actor')
        if tuple(policy.observation_space.shape) != (1544,) or policy.action_space.n != 90:
            raise ValueError('unexpected original actor contract')
        self.policy, self.variant = policy, variant
        self.horizon = 5 if variant in ('C','D') else 1
        self.context = HistoryContext() if variant != 'A' else None
        self.future = nn.ModuleList([deepcopy(policy.action_net) for _ in range(self.horizon - 1)])

    def logits(self, batch, context_enabled=True):
        feature = self.policy.extract_features(batch['observations'])
        if self.context is not None and context_enabled:
            residual = self.context(batch['recent'], batch['distant'], batch['inputs'])
            if 'context_known' in batch:
                residual = residual * batch['context_known'].reshape(-1, 1)
            feature = feature + residual
        latent = self.policy.mlp_extractor.forward_actor(feature)
        heads = [self.policy.action_net, *self.future]
        return torch.stack([head(latent).masked_fill(self.policy._blocked_actions, float('-inf'))
                            for head in heads], dim=1)

    def save(self, model, directory, name):
        directory = Path(directory)
        model.save(directory/f'{name}-actor.zip')
        torch.save(dict(context=self.context.state_dict() if self.context else None,
                        future=self.future.state_dict()), directory/f'{name}-context.pt')
        save_json(directory/f'{name}.json',dict(format=FORMAT, variant=self.variant,
            horizon=self.horizon,decision_frames=1,observation_shape=[1544],num_actions=90,
            recent_offsets=[20,15,10,5,0],distant_offsets=[60,45,30],own_input_offsets=[-60,-1],
            actor=f'{name}-actor.zip',actor_sha256=sha256(directory/f'{name}-actor.zip'),
            context=f'{name}-context.pt',context_sha256=sha256(directory/f'{name}-context.pt'),
            cold_start='Context residual disabled until60 consecutive own-history steps exist.',
            limitation='Actor ZIP alone omits B/C context;use this typed manifest. Not a PPO continuation artifact.'))


def load_actor(path, device):
    from stable_baselines3 import PPO
    path = Path(path).resolve(strict=True)
    metadata = json.loads(path.read_text())
    if (metadata.get('format') != FORMAT or metadata['decision_frames'] != 1
            or metadata['observation_shape'] != [1544] or metadata['num_actions'] != 90
            or metadata['recent_offsets'] != [20,15,10,5,0]
            or metadata['distant_offsets'] != [60,45,30]
            or metadata['own_input_offsets'] != [-60,-1]
            or metadata['horizon'] != (5 if metadata['variant'] in ('C','D') else 1)):
        raise ValueError('BC context manifest contract differs')
    for key in ('actor','context'):
        target = (path.parent/metadata[key]).resolve(strict=True)
        if not target.is_relative_to(path.parent) or sha256(target) != metadata[key+'_sha256']:
            raise ValueError('BC artifact identity changed')
    model = PPO.load(path.parent/metadata['actor'],device=device)
    actor = SequenceActor(model.policy, metadata['variant']).to(device)
    saved = torch.load(path.parent/metadata['context'],map_location=device,weights_only=True)
    if actor.context:
        actor.context.load_state_dict(saved['context'],strict=True)
    elif saved['context'] is not None:
        raise ValueError('legacy A must not load hidden context')
    actor.future.load_state_dict(saved['future'],strict=True)
    actor.eval()
    return model, actor, metadata


def tensor_batch(data, device):
    return {key:torch.as_tensor(data[key],device=device,dtype=torch.float32)
            for key in ('observations','recent','distant','inputs')}


def horizon_losses(logits, labels, mask, changed, change_weight=1.):
    if (logits.ndim != 3 or labels.shape != logits.shape[:2]
            or mask.shape != labels.shape or changed.shape != labels.shape
            or not math.isfinite(change_weight) or change_weight <= 0
            or torch.any(mask & ((labels < 0) | (labels >= logits.shape[-1])))):
        raise ValueError('invalid sequence targets,masks or change weight')
    losses = []
    for index in range(logits.shape[1]):
        selected = mask[:,index]
        if not selected.any():
            # Do not touch masked -inf card logits or invent a neutral target.
            losses.append(logits.new_zeros(()))
            continue
        values = nn.functional.cross_entropy(logits[selected,index],labels[selected,index],reduction='none')
        weights = torch.where(changed[selected,index],float(change_weight),1.)
        losses.append((values*weights).sum()/weights.sum())
    return torch.stack(losses)


def sequence_loss(logits, labels, mask, changed, change_weight=1., future_weight=None):
    if future_weight is not None:
        if (not math.isfinite(future_weight) or future_weight < 0
                or logits.shape[1] != 5 or not mask[:,0].any()):
            raise ValueError('weighted C requires5 horizons,current labels and finite nonnegative lambda')
        losses = horizon_losses(logits,labels,mask,changed,change_weight)
        return losses[0]+future_weight/4*losses[1:].sum()
    # Explicit legacy pooled reduction retains old experiments and A/B behavior.
    # A missing label never enters CE, even if its stored value is -1.
    selected = mask & (labels >= 0)
    if not selected.any():
        raise ValueError('batch has no valid supervision')
    values = nn.functional.cross_entropy(logits[selected],labels[selected],reduction='none')
    weights = torch.where(changed[selected],float(change_weight),1.)
    return (values*weights).sum()/weights.sum()


@torch.inference_mode()
def score(actor, dataset, device, batch_size=256, intervention=None):
    actor.eval()
    h = actor.horizon
    totals = np.zeros((h, 6), np.float64)
    exact = complete = 0
    predicted, labels_count = np.zeros((h,90),np.int64), np.zeros((h,90),np.int64)
    for start in range(0,dataset.count,batch_size):
        indices = np.arange(start,min(start+batch_size,dataset.count))
        data = dataset.batch(indices)
        batch = tensor_batch(data,device)
        if intervention:
            other = dataset.batch((indices+dataset.count//2) % dataset.count)
            for key in intervention:
                batch[key] = torch.as_tensor(other[key],device=device,dtype=torch.float32)
        logits = actor.logits(batch)
        truth = torch.as_tensor(data['labels'][:,:h],device=device,dtype=torch.long)
        mask = torch.as_tensor(data['mask'][:,:h],device=device,dtype=torch.bool)
        changed = torch.as_tensor(data['changed'][:,:h],device=device,dtype=torch.bool) & mask
        safe = truth.clamp_min(0)
        nll = -logits.log_softmax(-1).gather(-1,safe[...,None]).squeeze(-1)
        mode = logits.argmax(-1)
        for j in range(h):
            m,c = mask[:,j],changed[:,j]
            if not torch.isfinite(nll[m,j]).all():
                raise FloatingPointError('nonfinite sequence likelihood')
            totals[j] += [int(m.sum()),float(nll[m,j].sum()),int(((mode[:,j]==truth[:,j])&m).sum()),
                          int(c.sum()),float(nll[c,j].sum()),int(((mode[:,j]==truth[:,j])&c).sum())]
            predicted[j] += np.bincount(mode[m,j].cpu().numpy(),minlength=90)
            labels_count[j] += np.bincount(truth[m,j].cpu().numpy(),minlength=90)
        full = mask.all(-1)
        complete += int(full.sum())
        exact += int(((mode==truth).all(-1)&full).sum())
    return dict(horizons=[dict(offset=j,samples=int(v[0]),nll=v[1]/v[0],accuracy=v[2]/v[0],
        changed_samples=int(v[3]),changed_nll=v[4]/v[3] if v[3] else None,
        changed_accuracy=v[5]/v[3] if v[3] else None,
        predicted_counts=predicted[j].tolist(),label_counts=labels_count[j].tolist())
        for j,v in enumerate(totals)],complete_sequences=complete,
        sequence_exact_accuracy=exact/complete if complete else None)

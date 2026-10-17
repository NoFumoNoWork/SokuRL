"""Versioned semantic choice heads; unchanged v1 spatial/event backbone."""
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from soku_rl.rl.hierarchical_bc import HierarchicalActor, load_hierarchical
from soku_rl.replay.human_dataset import sha256, save_json


CATEGORIES = ('locomotion', 'flight_dash', 'melee', 'shot', 'skill', 'charged_shot')


def category_ids(catalog):
    result = []
    for item in catalog:
        if item['continuous']:
            kind = 'flight_dash' if item['buttons'] == 'D' else 'locomotion'
        else:
            command = item['command']
            kind = ('charged_shot' if '{' in command else 'skill' if len(command) > 2
                    else 'melee' if command[-1] == 'A' else 'shot')
        result.append(CATEGORIES.index(kind))
    if len(set(c['name'] for c in catalog)) != len(catalog):
        raise ValueError('duplicate command mapping')
    return result


class ChoiceHead(nn.Module):
    """P(action|state) = P(category|state) P(action|category,state)."""
    def __init__(self, catalog):
        super().__init__()
        self.category = nn.Linear(128, len(CATEGORIES))
        self.conditional = nn.Linear(128, len(catalog))
        self.register_buffer('mapping', torch.tensor(category_ids(catalog)))

    def forward(self, hidden, candidates):
        if candidates.shape != (len(hidden), len(self.mapping)) or not candidates.any(-1).all():
            raise ValueError('empty/invalid action candidate mask')
        local = self.conditional(hidden)
        category_mask = torch.stack([candidates[:, self.mapping == c].any(-1)
                                     for c in range(len(CATEGORIES))], -1)
        category = self.category(hidden).masked_fill(~category_mask, float('-inf'))
        log_category = category.log_softmax(-1)
        # An empty category is never normalized as an all -inf vector.
        conditional = torch.full_like(local, float('-inf'))
        for c in range(len(CATEGORIES)):
            members = self.mapping == c
            mask = candidates[:, members]
            values = local[:, members].masked_fill(~mask, float('-inf'))
            safe = torch.where(category_mask[:, c, None], values, torch.zeros_like(values))
            conditional[:, members] = safe.log_softmax(-1).masked_fill(~mask, float('-inf'))
        joint = conditional + log_category[:, self.mapping]
        return joint, category, conditional


class FactorizedActor(HierarchicalActor):
    event_history_version = 2

    def __init__(self, spatial_source, catalog):
        super().__init__(spatial_source, len(catalog))
        del self.action_head, self.sequence_head
        self.choice = ChoiceHead(catalog)
        self.future_choice = ChoiceHead(catalog)
        # Extra feedback kind (hit/block/resolution) fits v1's unused IDs 5..7.
        self.catalog = catalog

    def forward(self, batch, *, selected=None, teacher=None):
        context = self.context(batch)
        joint, category, conditional = self.choice(context, batch['candidates'].bool())
        active = batch['active_action'].long()
        elapsed = torch.log1p(batch['elapsed'].clamp_min(0))[:, None] / 8
        stop = self.stop_head(torch.cat((context, self.action_embedding(active), elapsed), -1)).squeeze(-1)
        chosen_category = category.argmax(-1)
        choice_id = conditional.masked_fill(
            self.choice.mapping[None] != chosen_category[:, None], float('-inf')).argmax(-1)
        previous = choice_id if selected is None else selected.long()
        if teacher is not None and teacher.shape != (len(context), 3):
            raise ValueError('three ordered future events required')
        hidden = context
        future = []
        # Current engine legality does not predict legality at future unknown states.
        future_mask = batch.get('future_candidates', torch.ones(
            len(context), 3, self.actions, dtype=torch.bool, device=context.device))
        if future_mask.shape != (len(context), 3, self.actions):
            raise ValueError('future mask shape mismatch')
        for i in range(3):
            hidden = self.sequence_cell(self.action_embedding(previous), hidden)
            step, future_category, future_conditional = self.future_choice(hidden, future_mask[:, i].bool())
            future.append(step)
            previous = (teacher[:, i].clamp_min(0).long() if teacher is not None else
                future_conditional.masked_fill(self.future_choice.mapping[None] !=
                    future_category.argmax(-1)[:, None], float('-inf')).argmax(-1))
        return dict(action=joint, category=category, conditional=conditional,
                    choice_id=choice_id, termination=stop, sequence=torch.stack(future, 1))


def rare_weights(labels, mask, actions):
    counts = torch.bincount(labels[mask.bool()].long(), minlength=actions).float()
    if not counts.any():
        raise ValueError('no training choice labels')
    reference = counts[counts > 0].max()
    return torch.where(counts > 0, (reference / counts.clamp_min(1)).sqrt().clamp(1, 4),
                       torch.ones_like(counts))


def factorized_loss(outputs, targets, weights):
    if not torch.isfinite(weights).all() or (weights < 1).any() or (weights > 4).any():
        raise ValueError('rare-label weights must be finite and bounded 1..4')

    def ce(logp, labels, mask):
        mask = mask.bool()
        if not mask.any():
            return logp[torch.isfinite(logp)].sum() * 0
        values = F.nll_loss(logp[mask], labels[mask].long(), reduction='none')
        if not torch.isfinite(values).all():
            raise ValueError('valid label excluded by action/category mask')
        # Denominator is valid labels, NOT all rows or sum of class weights.
        return (values * weights[labels[mask].long()]).sum() / mask.sum()

    choice = ce(outputs['action'], targets['action'], targets['action_mask'])
    future = ce(outputs['sequence'], targets['sequence'], targets['sequence_mask'])
    mask = targets['termination_mask'].bool()
    term = (F.binary_cross_entropy_with_logits(outputs['termination'][mask],
            targets['termination'][mask].float(), reduction='sum') / mask.sum()
            if mask.any() else outputs['termination'].sum() * 0)
    return dict(total=choice + .3 * term + .15 * future, choice=choice, termination=term, future=future)


def warm_start(actor, previous):
    old = previous.state_dict()
    target = actor.state_dict()
    copied = [k for k in target if k in old and target[k].shape == old[k].shape]
    target.update({k: old[k] for k in copied})
    actor.load_state_dict(target, strict=True)
    return dict(copied=copied, initialized=[k for k in target if k not in copied],
                discarded=[k for k in old if k not in copied], optimizer='FRESH')


def save_factorized(actor, path, source, metadata):
    path = Path(path)
    weights = path.with_suffix('.pt')
    if path.exists() or weights.exists():
        raise FileExistsError('preserve all previous checkpoints')
    torch.save(actor.state_dict(), weights)
    save_json(path, dict(format='hierarchical_bc_v2_factorized', catalog=actor.catalog,
        categories=CATEGORIES, mapping=category_ids(actor.catalog), source=str(Path(source).resolve()),
        source_sha256=sha256(source), weights=weights.name, weights_sha256=sha256(weights), metadata=metadata))


def load_factorized(path, device='cpu'):
    from stable_baselines3 import PPO
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text())
    if (data['format'] != 'hierarchical_bc_v2_factorized' or tuple(data['categories']) != CATEGORIES
            or data['mapping'] != category_ids(data['catalog'])):
        raise ValueError('factorized model contract mismatch')
    weights = (path.parent / data['weights']).resolve(strict=True)
    if (weights.parent != path.parent or sha256(weights) != data['weights_sha256']
            or sha256(data['source']) != data['source_sha256']):
        raise ValueError('checkpoint source/weights identity mismatch')
    original = PPO.load(data['source'], device='cpu')
    actor = FactorizedActor(original.policy.features_extractor, data['catalog']).to(device)
    actor.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    return actor.eval(), data


def load_event_actor(path, device='cpu'):
    kind = json.loads(Path(path).read_text())['format']
    if kind == 'hierarchical_bc_v1':
        return load_hierarchical(path, device)
    if kind == 'hierarchical_bc_v2_factorized':
        return load_factorized(path, device)
    raise ValueError('unknown event policy format')

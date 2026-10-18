"""Masked expert CE in the same optimizer step as ordinary feedforward PPO.

The clipped objective follows installed SB3 2.9 PPO.train, rather than the old
post-update online_teacher optimizer. Actions/rewards/GAE remain upstream's.
"""
import copy
import math
from importlib.metadata import version
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance

from soku_rl.policy.god.query import QueryTeacher, validate_spec


class ExpertSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    expert_actions: torch.Tensor
    expert_mask: torch.Tensor


class ExpertRolloutBuffer(RolloutBuffer):
    def reset(self):
        super().reset()
        self.expert_actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.expert_mask = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if not hasattr(self, 'query'):
            raise RuntimeError('kickstarting buffer requires a verified expert attachment')
        labels, valid = self.query(obs, episode_start, action)
        self.expert_actions[self.pos], self.expert_mask[self.pos] = labels, valid
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size=None):
        if not self.generator_ready:
            self.expert_actions = self.swap_and_flatten(self.expert_actions[..., None]).flatten()
            self.expert_mask = self.swap_and_flatten(self.expert_mask[..., None]).flatten()
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        base = super()._get_samples(batch_inds, env)
        return ExpertSamples(*base, self.to_torch(self.expert_actions[batch_inds]),
                             self.to_torch(self.expert_mask[batch_inds]))


def validate_kickstarting(config):
    if not isinstance(config, dict) or set(config) != {'teacher', 'lambda_initial', 'decay_transitions'}:
        raise ValueError('kickstarting requires teacher, lambda_initial and decay_transitions')
    validate_spec(config['teacher'])
    weight = config['lambda_initial']
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError('kickstarting lambda_initial must be finite and nonnegative')
    if type(config['decay_transitions']) is not int or config['decay_transitions'] < 1:
        raise ValueError('kickstarting decay_transitions must be positive')


def expert_weight(config, elapsed):
    return config['lambda_initial'] * max(0., 1. - max(0, elapsed) / config['decay_transitions'])


def expert_loss(policy, observations, labels, mask, zero):
    if not mask.any():
        return zero * 0.
    log_prob = policy.get_distribution(observations[mask]).log_prob(labels[mask])
    if not torch.isfinite(log_prob).all():
        raise RuntimeError('expert label conflicts with the student policy mask')
    return -log_prob.mean()


def attach_kickstarting(model, interface, config, continuing):
    validate_kickstarting(config)
    if config['teacher'].get('mode') == 'intent':
        raise RuntimeError('BLOCKED: intent queries require a trusted decision-boundary provider '
                           'and student-prefix-grounded 90-action label adapter; neither is wired to PPO')
    if not version('stable-baselines3').startswith('2.9.'):
        raise ValueError('joint-loss adapter requires the validated SB3 2.9.x PPO contract')
    if (not isinstance(model, KickstartingPPO) or type(model.rollout_buffer) is not ExpertRolloutBuffer
            or hasattr(model, '_expert')):
        raise ValueError('kickstarting requires its own unmodified PPO/buffer and one attachment')
    if not getattr(model.policy, 'masked_actions', ()):
        raise ValueError('kickstarting requires the existing static card-masked policy')
    expected = tuple(i for i, command in enumerate(interface.commands) if command & 48)
    if tuple(model.policy.masked_actions) != expected:
        raise ValueError('student and expert card masks must agree')
    teacher = QueryTeacher(interface, config['teacher'])
    identity = {'config': copy.deepcopy(config), 'teacher_fingerprint': teacher.fingerprint,
                'commands': list(interface.commands), 'loss': 'joint_ppo_masked_expert_ce_v1',
                'control': 'student_only', 'lambda_clock': 'collected_1F_transitions_at_rollout_start'}
    if continuing:
        if getattr(model, 'kickstarting_state', {}).get('identity') != identity:
            raise ValueError('continued kickstarting requires identical expert, clock and contracts')
    else:
        model.kickstarting_state = {'identity': identity, 'elapsed': 0, 'queries': 0,
            'valid': 0, 'matches': 0, 'statuses': {}, 'examples': [], 'episodes': 0}
    # A model checkpoint does not save a live native environment; actors restart
    # only at the next observed environment reset. Clock and seeds remain saved.
    model._expert = teacher
    model._expert_actors = [None] * model.n_envs
    model.rollout_buffer.query = model.query_expert


class KickstartingPPO(PPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['_expert', '_expert_actors']

    def query_expert(self, observations, starts, actions):
        if not hasattr(self, '_expert'):
            raise RuntimeError('kickstarting requires attachment via shared PPO factory')
        state = self.kickstarting_state
        labels, valid = np.zeros(self.n_envs, np.int64), np.zeros(self.n_envs, bool)
        for slot in range(self.n_envs):
            if starts[slot]:
                seed = (self._expert.spec['seed'] + state['episodes']) % 2**32
                self._expert_actors[slot] = self._expert.spawn(seed)
                state['episodes'] += 1
            actor = self._expert_actors[slot]
            if actor is None:
                raise RuntimeError('expert must begin at a genuine environment reset')
            view = observations[slot].view()
            view.flags.writeable = False
            advice = actor.query(view)
            valid[slot] = advice.status == 'VALID_INPUT' and advice.input_legal
            if valid[slot]:
                labels[slot] = advice.action
                state['valid'] += 1
                state['matches'] += int(labels[slot] == np.asarray(actions).reshape(-1)[slot])
            state['statuses'][advice.status] = state['statuses'].get(advice.status, 0) + 1
            if len(state['examples']) < 32:
                state['examples'].append(advice.record())
        state['queries'] += self.n_envs
        state['elapsed'] += self.n_envs
        return labels, valid

    def train(self):
        if not hasattr(self, '_expert'):
            raise RuntimeError('kickstarting training requires a verified expert attachment')
        cfg = self.kickstarting_state['identity']['config']
        start = self.kickstarting_state['elapsed'] - self.rollout_buffer.buffer_size * self.n_envs
        weight = expert_weight(cfg, start)
        if weight and not self.rollout_buffer.expert_mask.any():
            raise RuntimeError('BLOCKED: rollout has no reliable expert labels; do not train a fake teacher')
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_vf = None if self.clip_range_vf is None else self.clip_range_vf(self._current_progress_remaining)
        totals = {key: [] for key in ('entropy', 'policy_gradient', 'value', 'clip_fraction', 'expert_ce')}
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kls = []
            for data in self.rollout_buffer.get(self.batch_size):
                actions = data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(data.observations, actions)
                values = values.flatten()
                advantages = data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - data.old_log_prob)
                pg = -torch.min(advantages * ratio,
                    advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)).mean()
                predicted = values if clip_vf is None else data.old_values + torch.clamp(
                    values - data.old_values, -clip_vf, clip_vf)
                vf = F.mse_loss(data.returns, predicted)
                ent = -torch.mean(-log_prob) if entropy is None else -entropy.mean()
                ce = expert_loss(self.policy, data.observations, data.expert_actions,
                                 data.expert_mask, pg) if weight else pg * 0.
                loss = pg + self.ent_coef * ent + self.vf_coef * vf + weight * ce
                if not torch.isfinite(loss):
                    raise RuntimeError('non-finite kickstarting objective')
                for key, value in (('entropy', ent), ('policy_gradient', pg), ('value', vf),
                    ('clip_fraction', ((ratio-1).abs() > clip_range).float().mean()), ('expert_ce', ce)):
                    totals[key].append(float(value.detach()))
                with torch.no_grad():
                    log_ratio = log_prob - data.old_log_prob
                    kl = torch.mean((torch.exp(log_ratio)-1)-log_ratio).cpu().numpy()
                    approx_kls.append(kl)
                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm,
                                               error_if_nonfinite=True)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        for key in ('entropy', 'policy_gradient', 'value'):
            self.logger.record('train/'+key+'_loss', np.mean(totals[key]))
        self.logger.record('train/clip_fraction', np.mean(totals['clip_fraction']))
        self.logger.record('train/approx_kl', np.mean(approx_kls))
        self.logger.record('train/loss', loss.item())
        self.logger.record('train/explained_variance', explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/clip_range', clip_range)
        if clip_vf is not None:
            self.logger.record('train/clip_range_vf', clip_vf)
        state = self.kickstarting_state
        metrics = {'lambda': weight, 'expert_ce': float(np.mean(totals['expert_ce'])),
            'queries': state['queries'], 'valid': state['valid'], 'elapsed': state['elapsed'],
            'coverage': state['valid']/max(1, state['queries']),
            'behavior_agreement': state['matches']/max(1, state['valid'])}
        state['last_update'] = metrics
        for key, value in metrics.items():
            self.logger.record('kickstarting/'+key, value)

"""Bounded KL updates toward a frozen policy on the learner's actual online states."""
import copy
import hashlib
import io
import math
from pathlib import Path
import time

from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from soku_rl.policy.contract import read_training_contract
from soku_rl.rl.actor_windows import actor_window_logits, categorical_distance
from soku_rl.rl.storage import PackedObservation


def validate_anchor(config):
    keys = {'reference', 'updates_per_rollout', 'sequences', 'sequence_length', 'learning_rate', 'seed'}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError('online_anchor requires explicit reference, update budget, sequence dimensions, learning_rate and seed')
    reference = config['reference']
    if (not isinstance(reference, dict) or set(reference) != {'kind', 'path', 'training_config'}
            or reference['kind'] not in {'sb3', 'sb3_recurrent'}
            or any(not isinstance(reference[k], str) or not reference[k] for k in ('path', 'training_config'))):
        raise ValueError('online_anchor reference requires a frozen SB3 categorical checkpoint and training_config')
    for key in ('updates_per_rollout', 'sequences', 'sequence_length'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'online_anchor {key} must be a positive integer')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('online_anchor seed must be a nonnegative integer')
    rate = config['learning_rate']
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('online_anchor learning_rate must be finite and positive')


def attach_anchor(model, interface, config, continuing):
    from soku_rl.rl.ppo import parameter_hash
    validate_anchor(config)
    if hasattr(model, '_anchor'):
        raise ValueError('online_anchor is already attached to this learner')
    if (not isinstance(model.observation_space, spaces.Box)
            or model.observation_space.dtype != np.dtype('float32')
            or not isinstance(model.action_space, spaces.Discrete)):
        raise ValueError('online_anchor currently requires float32 Box observations and discrete actions')
    spec = config['reference']
    contract_path, path = Path(spec['training_config']), Path(spec['path'])
    contract_bytes, checkpoint_bytes = contract_path.read_bytes(), path.read_bytes()
    read_training_contract(contract_path, interface)
    if contract_path.read_bytes() != contract_bytes:
        raise RuntimeError('anchor reference contract changed during loading')
    algorithm = PPO if spec['kind'] == 'sb3' else RecurrentPPO
    # Loading constructs a new network. Do not consume the learner's action RNG
    # or reset Python/NumPy/CUDA seeds to those saved in the reference checkpoint.
    devices = [model.device.index] if model.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        reference = algorithm.load(io.BytesIO(checkpoint_bytes), device=model.device, custom_objects={'seed': None})
    if reference.observation_space != model.observation_space or reference.action_space != model.action_space:
        raise ValueError('anchor reference and learner spaces differ')
    reference.policy.set_training_mode(False)
    reference.policy.requires_grad_(False)
    identity = {'schema': 1, 'config': copy.deepcopy(config),
        'reference_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
        'reference_config_sha256': hashlib.sha256(contract_bytes).hexdigest(),
        'reference_parameter_hash': parameter_hash(reference.policy),
        'reference_steps': reference.num_timesteps, 'max_episode_steps': interface.episode.max_frames,
        'kl_direction': 'reference_to_current', 'optimizer': 'sgd_without_momentum'}
    if continuing:
        if not hasattr(model, 'anchor_state') or model.anchor_state['identity'] != identity:
            raise ValueError('continued online_anchor requires identical settings and frozen reference identity')
    else:
        model.anchor_state = {'identity': identity, 'rng': np.random.default_rng(config['seed']).bit_generator.state,
            'updates': 0, 'frames': 0, 'collected_frames': 0, 'current_burn_in_frames': 0, 'reference_burn_in_frames': 0}
    model._anchor = OnlinePolicyAnchor(model, reference, config, interface.episode.max_frames)


class OnlinePolicyAnchor:
    def __init__(self, model, reference, config, max_episode_steps):
        self.model, self.reference = model, reference
        self.config, self.max_episode_steps = copy.deepcopy(config), max_episode_steps
        self.active = [[] for _ in range(model.n_envs)]
        self.recent = []
        # No optimizer moments to serialize or borrow from the PPO update.
        self.optimizer = torch.optim.SGD(model.policy.parameters(), lr=config['learning_rate'],
                                         momentum=0., weight_decay=0.)
        buffer = model.rollout_buffer
        self.original_add, self.original_reset = buffer.add, buffer.reset
        # Observe the shared buffer boundary: upstream SB3 and joint MARL both
        # write here, without replacing sampling, GAE, or PPO's train method.
        buffer.add, buffer.reset = self.add, self.reset

    def reset(self):
        self.recent.clear()
        self.original_reset()

    def add(self, obs, action, reward, episode_start, value, log_prob, **kwargs):
        self.record(obs, episode_start)
        return self.original_add(obs, action, reward, episode_start, value, log_prob, **kwargs)

    def record(self, observations, starts):
        if (observations.shape != (len(self.active), *self.model.observation_space.shape)
                or observations.dtype != np.dtype('float32') or np.shape(starts) != (len(self.active),)):
            raise ValueError('anchor observation batch does not match its environments')
        for slot, start in enumerate(starts):
            if start:
                self.active[slot] = []
            episode = self.active[slot]
            if not start and not episode:
                raise RuntimeError('anchor requires an observed episode start and complete prefix')
            if len(episode) >= self.max_episode_steps:
                raise RuntimeError('anchor episode exceeds the configured game horizon')
            observation = PackedObservation.pack(observations[slot])
            self.recent.append((episode, len(episode)))
            episode.append(observation)
        self.model.anchor_state['collected_frames'] += len(self.active)

    def update(self):
        if not self.recent:
            raise RuntimeError('online_anchor requires a nonempty current rollout')
        started = time.perf_counter()
        model, config, state = self.model, self.config, self.model.anchor_state
        rng = np.random.default_rng()
        rng.bit_generator.state = state['rng']
        model.policy.set_training_mode(False)
        self.reference.policy.set_training_mode(False)
        frames, current_burn, reference_burn = 0, 0, 0
        totals = dict(kl_before=0., kl_after=0., total_variation_before=0., total_variation_after=0.)
        for _ in range(config['updates_per_rollout']):
            windows = []
            for index in rng.integers(len(self.recent), size=config['sequences']):
                episode, offset = self.recent[int(index)]
                length = min(config['sequence_length'], len(episode) - offset)
                target = actor_window_logits(self.reference, episode, offset, length, False)
                windows.append((episode, offset, length, target))
                reference_burn += offset if isinstance(self.reference, RecurrentPPO) else 0
            count = sum(row[2] for row in windows)
            self.optimizer.zero_grad(set_to_none=True)
            for episode, offset, length, target in windows:
                current = actor_window_logits(model, episode, offset, length, True)
                kl, distance = categorical_distance(target, current)
                loss = kl.sum() / count
                if not torch.isfinite(loss):
                    raise RuntimeError('non-finite online anchor KL loss')
                loss.backward()
                totals['kl_before'] += float(kl.detach().sum())
                totals['total_variation_before'] += float(distance.detach().sum())
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            for episode, offset, length, target in windows:
                current = actor_window_logits(model, episode, offset, length, False)
                kl, distance = categorical_distance(target, current)
                if not torch.isfinite(kl).all():
                    raise RuntimeError('non-finite post-update online anchor KL')
                totals['kl_after'] += float(kl.sum())
                totals['total_variation_after'] += float(distance.sum())
                current_burn += 2 * offset if isinstance(model, RecurrentPPO) else 0
            frames += count
            state['updates'] += 1
        state['rng'] = rng.bit_generator.state
        state['frames'] += frames
        state['current_burn_in_frames'] += current_burn
        state['reference_burn_in_frames'] += reference_burn
        metrics = {key: value / frames for key, value in totals.items()} | {
            'frames': frames, 'rollout_frames': len(self.recent), 'updates': state['updates'],
            'total_frames': state['frames'], 'collected_frames': state['collected_frames'],
            'current_burn_in_frames': current_burn, 'reference_burn_in_frames': reference_burn,
            'ppo_steps': model.num_timesteps, 'seconds': time.perf_counter() - started}
        state['last_update'] = metrics
        for key, value in metrics.items():
            model.logger.record(f'anchor/{key}', value)
        return metrics


class OnlineAnchorUpdates:
    def train(self):
        if not hasattr(self, '_anchor'):
            raise RuntimeError('online anchor training requires the shared PPO factory to attach its reference')
        super().train()
        self._anchor.update()

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['_anchor']


class AnchoredPPO(OnlineAnchorUpdates, PPO):
    pass


class AnchoredRecurrentPPO(OnlineAnchorUpdates, RecurrentPPO):
    pass


def anchored_algorithm(algorithm):
    if algorithm is PPO:
        return AnchoredPPO
    if algorithm is RecurrentPPO:
        return AnchoredRecurrentPPO
    raise ValueError('online anchor requires the shared PPO or RecurrentPPO implementation')

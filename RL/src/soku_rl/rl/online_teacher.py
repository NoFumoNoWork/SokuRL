"""Query a private rule teacher on learner states, then update the shared PPO actor."""
import copy
import math
import time

from gymnasium import spaces
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from soku_rl.policy.loader import load_policy
from soku_rl.rl.actor_windows import actor_window_logits
from soku_rl.rl.storage import PackedObservation
from soku_rl.rl.teacher_replay import load_teacher_replay, validate_teacher_replay


def validate_teacher(config):
    keys = {'teacher', 'updates_per_rollout', 'sequences', 'sequence_length', 'learning_rate', 'seed'}
    if not isinstance(config, dict) or set(config) not in (keys, keys | {'demonstration_replay'}):
        raise ValueError('online_teacher requires explicit teacher, update/window budgets, learning_rate and seed')
    spec = config['teacher']
    if (not isinstance(spec, dict) or set(spec) != {'kind', 'name', 'rules'}
            or spec['kind'] != 'rule' or not isinstance(spec['name'], str)
            or not spec['name'] or not isinstance(spec['rules'], dict)):
        raise ValueError('online_teacher requires an explicit frozen rule teacher')
    for key in ('updates_per_rollout', 'sequences', 'sequence_length'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'online_teacher {key} must be a positive integer')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('online_teacher seed must be a nonnegative integer')
    rate = config['learning_rate']
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('online_teacher learning_rate must be finite and positive')
    if 'demonstration_replay' in config:
        validate_teacher_replay(config['demonstration_replay'], config['sequences'])


def attach_teacher(model, interface, config, continuing):
    validate_teacher(config)
    if hasattr(model, '_online_teacher'):
        raise ValueError('online_teacher is already attached')
    if (not isinstance(model.observation_space, spaces.Box)
            or model.observation_space.dtype != np.dtype('float32')
            or not isinstance(model.action_space, spaces.Discrete)):
        raise ValueError('online_teacher requires float32 Box observations and discrete actions')
    teacher = load_policy('online_teacher', config['teacher'], interface, model.device)
    identity = {'schema': 1, 'config': copy.deepcopy(config), 'teacher_fingerprint': teacher.fingerprint,
        'max_episode_steps': interface.episode.max_frames, 'loss': 'sampled_rule_action_cross_entropy',
        'control': 'learner_only', 'optimizer': 'sgd_without_momentum'}
    if 'demonstration_replay' in config:
        replay, identities = load_teacher_replay(config['demonstration_replay'], interface, teacher.fingerprint)
        identity['demonstration_replay_datasets'] = identities
    if continuing:
        if not hasattr(model, 'teacher_state') or model.teacher_state['identity'] != identity:
            raise ValueError('continued online_teacher requires identical settings and teacher identity')
    else:
        rngs = {name: np.random.default_rng(np.random.SeedSequence(config['seed'], spawn_key=(i,))).bit_generator.state
                for i, name in enumerate(('teacher_rng', 'window_rng'))}
        model.teacher_state = {'identity': identity, **rngs, 'updates': 0, 'frames': 0,
            'collected_frames': 0, 'teacher_episodes': 0, 'behavior_matches': 0,
            'query_seconds': 0., 'collection_seconds': 0., 'current_burn_in_frames': 0, 'episodes': [],
            'label_counts': [0] * int(model.action_space.n)}
        if 'demonstration_replay' in config:
            model.teacher_state['demonstration_replay'] = {
                'rng': np.random.default_rng(config['demonstration_replay']['seed']).bit_generator.state,
                'updates': 0, 'frames': 0, 'burn_in_frames': 0}
    model._online_teacher = OnlineTeacher(model, teacher, config, interface.episode.max_frames)
    if 'demonstration_replay' in config:
        model._online_teacher.replay = replay


class OnlineTeacher:
    def __init__(self, model, teacher, config, max_episode_steps):
        self.model, self.teacher = model, teacher
        self.config, self.max_episode_steps = copy.deepcopy(config), max_episode_steps
        self.active, self.actors = [None] * model.n_envs, [None] * model.n_envs
        self.recent = []
        self.optimizer = torch.optim.SGD(model.policy.parameters(), lr=config['learning_rate'],
                                         momentum=0., weight_decay=0.)
        buffer = model.rollout_buffer
        self.original_add, self.original_reset = buffer.add, buffer.reset
        buffer.add, buffer.reset = self.add, self.reset

    def reset(self):
        self.recent.clear()
        self.original_reset()

    def add(self, obs, action, reward, episode_start, value, log_prob, **kwargs):
        self.record(obs, episode_start, action)
        return self.original_add(obs, action, reward, episode_start, value, log_prob, **kwargs)

    def record(self, observations, starts, actions):
        if (observations.shape != (len(self.active), *self.model.observation_space.shape)
                or observations.dtype != np.dtype('float32') or np.shape(starts) != (len(self.active),)
                or np.shape(actions) not in ((len(self.active),), (len(self.active), 1))):
            raise ValueError('online_teacher observation/action batch does not match its environments')
        started = time.perf_counter()
        state = self.model.teacher_state
        rng = np.random.default_rng()
        rng.bit_generator.state = state['teacher_rng']
        for slot, start in enumerate(starts):
            query_started = time.perf_counter()
            if start:
                seed = int(rng.integers(0, 0xFFFFFFFF))
                self.actors[slot] = self.teacher.spawn(seed)
                record = {'slot': slot, 'teacher_seed': seed,
                    'first_query': state['collected_frames'] + slot, 'frames': 0}
                state['episodes'].append(record)
                self.active[slot] = {'observations': [], 'labels': [], 'teacher_seed': seed, 'record': record}
                state['teacher_episodes'] += 1
            episode = self.active[slot]
            if episode is None:
                raise RuntimeError('online_teacher requires an observed episode start and complete prefix')
            if len(episode['observations']) >= self.max_episode_steps:
                raise RuntimeError('online_teacher episode exceeds the configured horizon')
            # Teachers see the same observation and actual submitted-action
            # history as the learner, without changing it or controlling play.
            observation = observations[slot].view()
            observation.flags.writeable = False
            label = self.actors[slot].act(observation)
            state['query_seconds'] += time.perf_counter() - query_started
            if isinstance(label, (bool, np.bool_)) or not self.model.action_space.contains(label):
                raise ValueError('online teacher action is outside the learning vocabulary')
            label = int(label)
            self.recent.append((episode, len(episode['observations'])))
            episode['observations'].append(PackedObservation.pack(observation))
            episode['labels'].append(label)
            episode['record']['frames'] += 1
            state['label_counts'][label] += 1
            state['behavior_matches'] += int(label == np.asarray(actions).reshape(-1)[slot])
        state['teacher_rng'] = rng.bit_generator.state
        state['collected_frames'] += len(self.active)
        state['collection_seconds'] += time.perf_counter() - started

    def update(self):
        if not self.recent:
            raise RuntimeError('online_teacher requires a nonempty current rollout')
        started = time.perf_counter()
        model, config, state = self.model, self.config, self.model.teacher_state
        rng = np.random.default_rng()
        rng.bit_generator.state = state['window_rng']
        model.policy.set_training_mode(False)
        frames, burn = 0, 0
        gradient_norms = []
        totals = dict(nll_before=0., nll_after=0., accuracy_before=0., accuracy_after=0.)
        grouped = {name: dict(totals, frames=0, burn_in_frames=0) for name in ('online', 'replay')}
        replay_sequences = 0
        if 'demonstration_replay' in config:
            replay_sequences = config['demonstration_replay']['sequences']
            replay_rng = np.random.default_rng()
            replay_rng.bit_generator.state = state['demonstration_replay']['rng']
        for _ in range(config['updates_per_rollout']):
            windows = []
            for index in rng.integers(len(self.recent), size=config['sequences'] - replay_sequences):
                episode, offset = self.recent[int(index)]
                length = min(config['sequence_length'], len(episode['observations']) - offset)
                target = torch.tensor(episode['labels'][offset:offset + length], device=model.device)
                windows.append((episode['observations'], offset, length, target, 'online'))
            if replay_sequences:
                windows.extend(self.replay.sample(replay_rng, replay_sequences, config['sequence_length'], model.device))
            count = sum(row[2] for row in windows)
            self.optimizer.zero_grad(set_to_none=True)
            for episode, offset, length, target, source in windows:
                logits = actor_window_logits(model, episode, offset, length, True)
                nll = F.cross_entropy(logits, target, reduction='sum')
                if not torch.isfinite(nll):
                    raise RuntimeError('non-finite online teacher cross entropy')
                (nll / count).backward()
                nll_value, correct = float(nll.detach()), int((logits.argmax(-1) == target).sum())
                totals['nll_before'] += nll_value
                totals['accuracy_before'] += correct
                grouped[source]['nll_before'] += nll_value
                grouped[source]['accuracy_before'] += correct
                grouped[source]['frames'] += length
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.policy.parameters(), model.max_grad_norm, error_if_nonfinite=True)
            gradient_norms.append(float(gradient_norm))
            self.optimizer.step()
            for episode, offset, length, target, source in windows:
                logits = actor_window_logits(model, episode, offset, length, False)
                nll = F.cross_entropy(logits, target, reduction='sum')
                if not torch.isfinite(nll):
                    raise RuntimeError('non-finite post-update online teacher cross entropy')
                nll_value, correct = float(nll), int((logits.argmax(-1) == target).sum())
                totals['nll_after'] += nll_value
                totals['accuracy_after'] += correct
                grouped[source]['nll_after'] += nll_value
                grouped[source]['accuracy_after'] += correct
                grouped[source]['burn_in_frames'] += 2 * offset if isinstance(model, RecurrentPPO) else 0
                burn += 2 * offset if isinstance(model, RecurrentPPO) else 0
            frames += count
            state['updates'] += 1
        state['window_rng'] = rng.bit_generator.state
        state['frames'] += frames
        state['current_burn_in_frames'] += burn
        metrics = {key: value / frames for key, value in totals.items()} | {
            'frames': frames, 'rollout_frames': len(self.recent), 'updates': state['updates'],
            'total_frames': state['frames'], 'collected_frames': state['collected_frames'],
            'teacher_episodes': state['teacher_episodes'], 'query_seconds': state['query_seconds'],
            'collection_seconds': state['collection_seconds'],
            'behavior_agreement': state['behavior_matches'] / state['collected_frames'],
            'current_burn_in_frames': burn, 'ppo_steps': model.num_timesteps,
            'gradient_norm_mean': sum(gradient_norms) / len(gradient_norms),
            'gradient_norm_max': max(gradient_norms),
            'gradient_clip_fraction': sum(norm > model.max_grad_norm for norm in gradient_norms) / len(gradient_norms),
            'gradient_limit': model.max_grad_norm, 'learning_rate': config['learning_rate'],
            'seconds': time.perf_counter() - started}
        if replay_sequences:
            replay_state = state['demonstration_replay']
            replay_state['rng'] = replay_rng.bit_generator.state
            replay_state['updates'] += config['updates_per_rollout']
            replay_state['frames'] += grouped['replay']['frames']
            replay_state['burn_in_frames'] += grouped['replay']['burn_in_frames']
            for source, values in grouped.items():
                metrics.update({f'{source}_{key}': value / values['frames'] for key, value in values.items()
                                if key in totals})
                metrics[f'{source}_frames'] = values['frames']
                metrics[f'{source}_burn_in_frames'] = values['burn_in_frames']
        state['last_update'] = metrics
        for key, value in metrics.items():
            model.logger.record(f'teacher/{key}', value)
        return metrics


class OnlineTeacherUpdates:
    def train(self):
        if not hasattr(self, '_online_teacher'):
            raise RuntimeError('online teacher training requires attachment through the shared PPO factory')
        super().train()
        self._online_teacher.update()

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['_online_teacher']


class TeacherPPO(OnlineTeacherUpdates, PPO):
    pass


class TeacherRecurrentPPO(OnlineTeacherUpdates, RecurrentPPO):
    pass


def teacher_algorithm(algorithm):
    if algorithm is PPO:
        return TeacherPPO
    if algorithm is RecurrentPPO:
        return TeacherRecurrentPPO
    raise ValueError('online teacher requires shared PPO or RecurrentPPO')

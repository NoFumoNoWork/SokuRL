"""Observe carried versus freshly replayed PPO memory without changing training."""
import hashlib
import json
from pathlib import Path
import time

from gymnasium import spaces
import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from soku_rl.rl.actor_windows import categorical_distance
from soku_rl.rl.ppo import parameter_hash
from soku_rl.rl.recurrent_cloning import zero_states
from soku_rl.rl.sparse_transfer import restore_batch
from soku_rl.rl.storage import PackedObservation


def minibatch_consistency(values, log_prob, batch):
    mask = batch.mask > 1e-8
    if not mask.any():
        raise ValueError('consistency probe requires unpadded samples')
    log_ratio = log_prob.detach()[mask] - batch.old_log_prob[mask]
    value_delta = values.detach().flatten()[mask] - batch.old_values[mask]
    if not torch.isfinite(log_ratio).all() or not torch.isfinite(value_delta).all():
        raise RuntimeError('non-finite unpadded minibatch consistency values')
    stable = log_ratio.double()
    return {'valid_samples': int(mask.sum()), 'padded_samples': int((~mask).sum()),
        'mean_abs_log_ratio': float(log_ratio.abs().mean()),
        'max_abs_log_ratio': float(log_ratio.abs().max()),
        'mean_abs_value_delta': float(value_delta.abs().mean()),
        'max_abs_value_delta': float(value_delta.abs().max()),
        'sample_approx_kl': float((torch.expm1(stable) - stable).mean()),
        'sb3_float32_approx_kl': float(((torch.exp(log_ratio) - 1) - log_ratio).mean())}


def attach_recurrent_state_diagnostic(model, max_episode_steps):
    if type(model) is not RecurrentPPO:
        raise ValueError('memory diagnostic requires the unmodified shared RecurrentPPO')
    if (not isinstance(model.observation_space, spaces.Box)
            or model.observation_space.dtype != np.dtype('float32')
            or not isinstance(model.action_space, spaces.Discrete)):
        raise ValueError('memory diagnostic requires float32 Box observations and discrete actions')
    if type(max_episode_steps) is not int or max_episode_steps < 1:
        raise ValueError('memory diagnostic requires a positive episode bound')
    buffer = model.rollout_buffer
    if any(not hasattr(getattr(buffer, name), '__self__')
            or getattr(buffer, name).__self__ is not buffer for name in ('add', 'reset', 'get')):
        raise ValueError('memory diagnostic requires an unwrapped rollout buffer')
    if (not hasattr(model.policy.evaluate_actions, '__self__')
            or model.policy.evaluate_actions.__self__ is not model.policy):
        raise ValueError('memory diagnostic requires unwrapped action evaluation')
    observer = RecurrentStateDiagnostic(model, max_episode_steps)
    # Upstream excludes rollout_buffer and serializes only policy tensors, not
    # instance methods. No observer or episode history enters the checkpoint.
    buffer.add, buffer.reset = observer.add, observer.reset
    buffer.get, model.policy.evaluate_actions = observer.get, observer.evaluate_actions
    return observer


class RecurrentStateDiagnostic:
    def __init__(self, model, max_episode_steps):
        self.model, self.max_episode_steps = model, max_episode_steps
        self.original_add = model.rollout_buffer.add
        self.original_reset = model.rollout_buffer.reset
        self.original_get = model.rollout_buffer.get
        self.original_evaluate_actions = model.policy.evaluate_actions
        self.active = [[] for _ in range(model.n_envs)]
        self.rollouts = []
        self.frames = []
        self.shadow = zero_states(model.policy, model.n_envs)
        self.boundary = {}
        self.observer_seconds = 0.

    def get(self, batch_size):
        for batch in self.original_get(batch_size):
            self.current_batch = batch
            try:
                yield batch
            finally:
                del self.current_batch

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        result = self.original_evaluate_actions(obs, actions, lstm_states, episode_starts)
        if not self.rollouts or not hasattr(self, 'current_batch'):
            raise RuntimeError('consistency probe requires a completed rollout and its current minibatch')
        row = self.rollouts[-1]
        if 'first_minibatch' not in row:
            batch = self.current_batch
            if (obs is not batch.observations or lstm_states is not batch.lstm_states
                    or episode_starts is not batch.episode_starts):
                raise RuntimeError('action evaluation inputs differ from the sampled minibatch')
            assert self.model._n_updates == row['boundary']['ppo_n_updates']
            with torch.no_grad():
                row['first_minibatch'] = minibatch_consistency(result[0], result[1], batch)
            row['first_minibatch']['ppo_n_updates_before'] = self.model._n_updates
            self._write_report()
        # Return the original tensors and their computation graphs unchanged.
        return result

    def _write_report(self):
        directory = self.model.logger.get_dir()
        if not directory:
            raise ValueError('memory diagnostic requires a persisted training logger directory')
        path = Path(directory).parent / 'recurrent-state-audit.json'
        path.write_text(json.dumps({'schema': 2,
            'method': 'shadow replay and pre-update minibatch consistency; no replacement of online memory',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rollouts': self.rollouts}, indent=2))

    def _replay(self):
        model = self.model
        states = zero_states(model.policy, model.n_envs)
        length = max(map(len, self.active))
        fallback = [PackedObservation.pack(row) for row in model._last_obs]
        with torch.no_grad():
            for index in range(length):
                observations, starts = [], []
                for slot, episode in enumerate(self.active):
                    offset = index - (length - len(episode))
                    observations.append(episode[offset] if offset >= 0 else fallback[slot])
                    starts.append(offset <= 0)
                # Same one-frame, n_envs-wide inference shape as actual sampling.
                batch = restore_batch(observations, model.device)
                _, _, _, states = model.policy(batch, states,
                    torch.tensor(starts, dtype=torch.float32, device=model.device), deterministic=True)
            for slot, episode in enumerate(self.active):
                if not episode:
                    for tensor in (*states.pi, *states.vf):
                        tensor[:, slot].zero_()
        return states

    def reset(self):
        started = time.perf_counter()
        self.original_reset()
        model = self.model
        if model.policy.training:
            raise RuntimeError('memory audit must run after the sampler switches policy to eval')
        if self.frames:
            raise RuntimeError('previous diagnostic rollout was not completed')
        for slot, start in enumerate(model._last_episode_starts):
            if start:
                self.active[slot] = []
            elif not self.active[slot]:
                raise RuntimeError('memory audit requires complete observed episode prefixes')
        before = parameter_hash(model.policy)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(model.device) if model.device.type == 'cuda' else []
        carried = [tensor.clone() for tensor in (*model._last_lstm_states.pi, *model._last_lstm_states.vf)]
        if any(self.active):
            distribution = model.policy.action_dist.distribution
            self.shadow = self._replay()
            model.policy.action_dist.distribution = distribution
        else:
            # Before the first policy call SB3 has not constructed a categorical
            # distribution yet. Empty histories need no inference or cache edit.
            self.shadow = zero_states(model.policy, model.n_envs)
        assert parameter_hash(model.policy) == before and torch.equal(cpu_rng, torch.get_rng_state())
        if model.device.type == 'cuda':
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(model.device))
        assert all(torch.equal(a, b) for a, b in zip(carried,
            (*model._last_lstm_states.pi, *model._last_lstm_states.vf), strict=True))
        distances = {}
        for name, old, fresh in zip(('actor_h', 'actor_c', 'critic_h', 'critic_c'), carried,
                (*self.shadow.pi, *self.shadow.vf), strict=True):
            for slot, start in enumerate(model._last_episode_starts):
                if start:
                    old[:, slot].zero_()
            distances[name] = ((old - fresh).square().mean(dim=(0, 2)).sqrt()).cpu().tolist()
        self.boundary = {'start_steps': model.num_timesteps, 'ppo_n_updates': model._n_updates,
            'parameter_hash': before, 'prefix_frames': list(map(len, self.active)),
            'episode_starts': model._last_episode_starts.astype(bool).tolist(),
            'state_rmse_by_slot': distances, 'replay_preserved_parameters_rng_and_carried_states': True}
        self.observer_seconds = time.perf_counter() - started

    def add(self, obs, action, reward, episode_start, value, log_prob, **kwargs):
        started = time.perf_counter()
        model = self.model
        if (obs.shape != (model.n_envs, *model.observation_space.shape)
                or obs.dtype != np.dtype('float32') or model.policy.training or not self.boundary):
            raise ValueError('memory diagnostic requires initialized float32 eval-mode rollouts')
        distribution = model.policy.action_dist.distribution
        actual_logits = distribution.logits.clone()
        with torch.no_grad():
            batch = torch.as_tensor(obs, device=model.device)
            _, shadow_values, _, self.shadow = model.policy(batch, self.shadow,
                torch.as_tensor(episode_start, dtype=torch.float32, device=model.device), deterministic=True)
            shadow_distribution = model.policy.action_dist.distribution
            shadow_logits = shadow_distribution.logits.clone()
            shadow_log_probs = shadow_distribution.log_prob(torch.as_tensor(action, device=model.device).flatten())
            kl, tv = categorical_distance(actual_logits, shadow_logits)
            metrics = torch.stack((kl, tv, shadow_values.flatten() - value.flatten(),
                shadow_log_probs - log_prob.flatten(),
                (actual_logits.argmax(-1) != shadow_logits.argmax(-1)).double()), dim=1).cpu().tolist()
        model.policy.action_dist.distribution = distribution
        for slot, start in enumerate(episode_start):
            if start:
                self.active[slot] = []
            if len(self.active[slot]) >= self.max_episode_steps:
                raise RuntimeError('diagnostic episode exceeds declared horizon')
            self.active[slot].append(PackedObservation.pack(obs[slot]))
            self.frames.append({'steps': model.num_timesteps, 'slot': slot,
                'rollout_frame': model.rollout_buffer.pos, 'episode_frame': len(self.active[slot]) - 1,
                **dict(zip(('kl', 'tv', 'value_delta', 'action_log_prob_delta', 'argmax_changed'),
                    metrics[slot], strict=True))})
        result = self.original_add(obs, action, reward, episode_start, value, log_prob, **kwargs)
        self.observer_seconds += time.perf_counter() - started
        if model.rollout_buffer.full:
            if len(self.frames) != model.n_steps * model.n_envs:
                raise RuntimeError('diagnostic frame count differs from completed rollout')
            self.rollouts.append({'boundary': self.boundary, 'end_steps': model.num_timesteps,
                'observer_seconds': self.observer_seconds, 'frames': self.frames})
            self.frames = []
            self._write_report()
        return result

import json
from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from soku_rl.rl.recurrent_state_diagnostics import attach_recurrent_state_diagnostic, minibatch_consistency


class MemoryEnv(gym.Env):
    def __init__(self, length):
        self.length = length
        self.observation_space = spaces.Box(-2, 2, (4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.steps = 0
        self.episodes = -1
        self.actions = []

    def observation(self):
        return np.array([self.steps / 7, self.episodes % 3 / 3, self.length / 7, 1.], dtype=np.float32)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.steps = 0
        self.episodes += 1
        return self.observation(), {}

    def step(self, action):
        self.actions.append(int(action))
        self.steps += 1
        return self.observation(), float(action == self.steps % 3), self.steps == self.length, False, {}


def create_model(device, directory):
    env = DummyVecEnv([lambda: MemoryEnv(3), lambda: MemoryEnv(7)])
    model = RecurrentPPO('MlpLstmPolicy', env, seed=731, device=device,
        n_steps=4, batch_size=8, n_epochs=2, learning_rate=.003,
        policy_kwargs={'lstm_hidden_size': 8, 'net_arch': {'pi': [8], 'vf': [8]}})
    model.set_logger(configure(str(directory / 'scalars'), []))
    return model


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        # SB3 loading may move Adam's scalar step tensor to the model device.
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_observer_preserves_actual_ppo_and_tracks_memory_drift(tmp_path, device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    control = create_model(device, tmp_path / 'control')
    control.learn(32)
    control_cpu_rng = torch.get_rng_state()
    control_cuda_rng = torch.cuda.get_rng_state(device) if device.startswith('cuda') else []
    observed = create_model(device, tmp_path / 'observed')
    observer = attach_recurrent_state_diagnostic(observed, 7)
    observed.learn(32)
    assert torch.equal(control_cpu_rng, torch.get_rng_state())
    if device.startswith('cuda'):
        assert torch.equal(control_cuda_rng, torch.cuda.get_rng_state(device))
    assert_tree_equal(control.policy.state_dict(), observed.policy.state_dict())
    assert_tree_equal(control.policy.optimizer.state_dict(), observed.policy.optimizer.state_dict())
    assert_tree_equal(control._last_lstm_states, observed._last_lstm_states)
    assert np.array_equal(control._last_obs, observed._last_obs)
    assert control._n_updates == observed._n_updates == 8
    for a, b in zip(control.env.envs, observed.env.envs, strict=True):
        assert a.actions == b.actions
    assert len(observer.rollouts) == 4 and not observer.frames
    assert all(row['tv'] == row['value_delta'] == 0 for row in observer.rollouts[0]['frames'])
    assert any(row['tv'] > 0 for rollout in observer.rollouts[1:] for row in rollout['frames'])
    starts = [row for rollout in observer.rollouts for row in rollout['frames'] if row['episode_frame'] == 0]
    assert len(starts) > 2 and all(row['tv'] == row['value_delta'] == 0 for row in starts)
    saved = json.loads((tmp_path / 'observed/recurrent-state-audit.json').read_text())
    assert saved['rollouts'] == observer.rollouts
    assert all(len(row['frames']) == 8 for row in saved['rollouts'])
    for row in saved['rollouts']:
        probe = row['first_minibatch']
        assert probe['valid_samples'] == 8
        assert probe['max_abs_log_ratio'] < 1e-5
        assert probe['max_abs_value_delta'] < 1e-5
        assert probe['ppo_n_updates_before'] == row['boundary']['ppo_n_updates']
    assert any(row['first_minibatch']['padded_samples'] > 0 for row in saved['rollouts'])
    checkpoint = tmp_path / 'observed/model.zip'
    observed.save(checkpoint)
    loaded = RecurrentPPO.load(checkpoint, device=device)
    assert_tree_equal(loaded.policy.state_dict(), observed.policy.state_dict())
    assert_tree_equal(loaded.policy.optimizer.state_dict(), observed.policy.optimizer.state_dict())
    assert loaded.rollout_buffer.add.__self__ is loaded.rollout_buffer
    assert loaded.policy.evaluate_actions.__self__ is loaded.policy
    control.env.close()
    observed.env.close()


@pytest.mark.parametrize('bound', [0, -1, True, 1.5])
def test_invalid_episode_bound(tmp_path, bound):
    model = create_model('cpu', tmp_path)
    with pytest.raises(ValueError, match='positive episode bound'):
        attach_recurrent_state_diagnostic(model, bound)
    model.env.close()


def test_minibatch_probe_excludes_padding_and_uses_stable_kl():
    values = torch.tensor([[.5], [999.]])
    log_probs = torch.tensor([-.4, -999.])
    batch = SimpleNamespace(mask=torch.tensor([1., 0.]), old_log_prob=torch.tensor([-.5, 123.]),
        old_values=torch.tensor([.25, 123.]))
    result = minibatch_consistency(values, log_probs, batch)
    assert result['valid_samples'] == result['padded_samples'] == 1
    assert result['max_abs_value_delta'] == .25
    assert result['max_abs_log_ratio'] == pytest.approx(.1)
    assert result['sample_approx_kl'] == pytest.approx(np.expm1(.1) - .1)
    batch.mask.zero_()
    with pytest.raises(ValueError, match='unpadded samples'):
        minibatch_consistency(values, log_probs, batch)


def test_rejects_duplicate_observer_and_missing_prefix(tmp_path):
    model = create_model('cpu', tmp_path)
    observer = attach_recurrent_state_diagnostic(model, 7)
    with pytest.raises(ValueError, match='unwrapped rollout buffer'):
        attach_recurrent_state_diagnostic(model, 7)
    model._last_obs = np.zeros((2, 4), dtype=np.float32)
    model._last_episode_starts = np.zeros(2, dtype=bool)
    model.policy.set_training_mode(False)
    with pytest.raises(RuntimeError, match='complete observed episode prefixes'):
        observer.reset()
    model.env.close()

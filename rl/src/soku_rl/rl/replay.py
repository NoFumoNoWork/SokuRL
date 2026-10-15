"""Lossless sparse storage with upstream SB3 n-step returns and boundary handling."""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import NStepReplayBuffer

from soku_rl.rl.storage import PackedArray
from soku_rl.rl.sparse_transfer import restore_batch


class ReplayObservations(PackedArray):
    """Keep samples packed until the upstream buffer calls to_torch."""
    def __getitem__(self, key):
        return self.samples[key]


class DictionaryObservations:
    def __init__(self, space, rows, n_envs):
        self.fields = {key: ReplayObservations(np.empty((rows, n_envs), dtype=object),
                                              value.shape, value.dtype) for key, value in space.spaces.items()}

    def __getitem__(self, key):
        return {name: field[key] for name, field in self.fields.items()}

    def __setitem__(self, key, values):
        if set(values) != set(self.fields):
            raise ValueError("replay dictionary fields changed")
        for name, field in self.fields.items():
            field[key] = values[name]


class PackedNStepReplayBuffer(NStepReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs,
                 optimize_memory_usage, n_steps, gamma, handle_timeout_termination):
        if not isinstance(observation_space, (spaces.Box, spaces.Dict)):
            raise TypeError("packed n-step replay requires Box or Dict observations")
        if optimize_memory_usage or handle_timeout_termination:
            raise ValueError("packed replay requires explicit next states and finite-horizon terminals")
        if buffer_size // n_envs < n_steps:
            raise ValueError("replay capacity per environment must be at least n_steps")
        # Avoid allocating the full padded observation tensor even temporarily.
        empty = spaces.Box(0, 1, shape=(0,), dtype=np.float32)
        super().__init__(buffer_size, empty, action_space, device=device, n_envs=n_envs,
            optimize_memory_usage=False, handle_timeout_termination=False, n_steps=n_steps, gamma=gamma)
        self.observation_space = observation_space
        self.obs_shape = observation_space.shape
        for name in ("observations", "next_observations"):
            if isinstance(observation_space, spaces.Dict):
                storage = DictionaryObservations(observation_space, self.buffer_size, n_envs)
            else:
                storage = ReplayObservations(np.empty((self.buffer_size, n_envs), dtype=object),
                                             self.obs_shape, observation_space.dtype)
            setattr(self, name, storage)

    def add(self, obs, next_obs, action, reward, done, infos):
        if not isinstance(self.observation_space, spaces.Dict):
            self.timeouts[self.pos] = 0
            return super().add(obs, next_obs, action, reward, done, infos)
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = np.asarray(action).reshape(self.n_envs, self.action_dim)
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.timeouts[self.pos] = 0
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full, self.pos = True, 0

    def _get_samples(self, batch_inds, env):
        if env is not None:
            raise ValueError("packed replay does not support VecNormalize")
        return super()._get_samples(batch_inds, env)

    def to_torch(self, array, *args, **kwargs):
        if isinstance(array, dict):
            return {key: self.to_torch(value, *args, **kwargs) for key, value in array.items()}
        if array.dtype == object:
            if np.dtype(array[0].dtype) == np.dtype("<f4"):
                return restore_batch(array, self.device)
            array = np.stack([sample.unpack() for sample in array])
        return super().to_torch(array, *args, **kwargs)

    def cut_trajectories(self):
        # Resumption starts new games. End the old n-step sequence at the last
        # recorded next state, bootstrapping there unless it was truly terminal.
        if self.pos or self.full:
            self.timeouts[self.pos - 1] = 1 - self.dones[self.pos - 1]

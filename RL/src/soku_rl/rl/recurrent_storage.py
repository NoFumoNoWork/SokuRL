"""Lossless observation storage with upstream recurrent sequencing and updates."""
from gymnasium import spaces
import numpy as np
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer

from soku_rl.rl.sparse_transfer import restore_batch
from soku_rl.rl.storage import PackedArray, PackedObservation, SPARSE_WORDS


class SparseRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    def reset(self):
        if (not isinstance(self.observation_space, spaces.Box)
                or self.observation_space.dtype != np.dtype('<f4')):
            raise ValueError('sparse recurrent storage requires a float32 Box observation')
        shape = self.obs_shape
        self.obs_shape = (0,)
        try:
            super().reset()
        finally:
            self.obs_shape = shape
        self.observations = PackedArray(np.empty((self.buffer_size, self.n_envs), dtype=object),
            shape, self.observation_space.dtype)

    def _get_samples(self, batch_inds, env_change, *normalizers):
        observations, shape = self.observations, self.obs_shape
        # Let upstream choose sequences, pad all other fields and select the
        # exact initial memories. An empty feature axis avoids dense host copies.
        self.observations = np.empty((len(observations.samples), 0), dtype=np.float32)
        self.obs_shape = (0,)
        try:
            batch = super()._get_samples(batch_inds, env_change, *normalizers)
        finally:
            self.observations, self.obs_shape = observations, shape
        starts = self.seq_start_indices
        padded_size = batch.observations.shape[0]
        length = padded_size // len(starts)
        empty = PackedObservation(shape, np.dtype('<f4').str, SPARSE_WORDS + bytes(4))
        padded = np.full(padded_size, empty, dtype=object)
        ends = np.concatenate((starts[1:], [len(batch_inds)]))
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            padded[index * length:index * length + end - start] = observations.samples[batch_inds[start:end]]
        return batch._replace(observations=restore_batch(padded, self.device))


def attach_sparse_recurrent_buffer(model):
    original = model.rollout_buffer
    if type(original) is not RecurrentRolloutBuffer or original.pos or original.full:
        raise ValueError('sparse recurrent storage requires an empty original recurrent buffer')
    model.rollout_buffer = SparseRecurrentRolloutBuffer(original.buffer_size,
        original.observation_space, original.action_space, original.hidden_state_shape,
        original.device, gae_lambda=original.gae_lambda, gamma=original.gamma, n_envs=original.n_envs)


def attach_requested_storage(model, config):
    if 'recurrent_storage' in config['ppo']:
        attach_sparse_recurrent_buffer(model)
    return model

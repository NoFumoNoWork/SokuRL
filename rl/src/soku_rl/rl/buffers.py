"""Change observation storage while retaining SB3 sampling, GAE and PPO updates."""
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer

from soku_rl.rl.storage import PackedArray


class PackedRolloutBuffer(RolloutBuffer):
    def reset(self):
        shape = self.obs_shape
        # Let SB3 allocate and initialize every training field. Allocate zero
        # dense observation elements, then use lossless per-sample storage.
        self.obs_shape = (0,)
        try:
            super().reset()
        finally:
            self.obs_shape = shape
        self.observations = PackedArray(np.empty((self.buffer_size, self.n_envs), dtype=object),
                                        shape, self.observation_space.dtype)

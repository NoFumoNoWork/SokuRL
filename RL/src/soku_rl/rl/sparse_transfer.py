"""Restore padded float32 observations on device after transferring nonzero words."""
import numpy as np
import torch
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from soku_rl.rl.buffers import PackedRolloutBuffer
from soku_rl.rl.storage import SPARSE_WORDS


def restore_batch(samples, device):
    if len(samples) == 0:
        raise ValueError("sparse transfer requires a nonempty observation batch")
    shape = samples[0].shape
    width = int(np.prod(shape))
    if any(sample.shape != shape or np.dtype(sample.dtype) != np.dtype("<f4") for sample in samples):
        raise ValueError("sparse transfer requires matching little-endian float32 observations")
    target = torch.zeros((len(samples), *shape), dtype=torch.float32, device=device)
    positions, values = [], []
    for row, sample in enumerate(samples):
        if not sample.content.startswith(SPARSE_WORDS):
            target[row].copy_(torch.from_numpy(sample.unpack()).to(device))
            continue
        count = int.from_bytes(sample.content[8:12], "little")
        if len(sample.content) != 12 + 8 * count or count > width:
            raise ValueError("invalid sparse observation header")
        indices = np.frombuffer(sample.content, dtype="<u4", count=count, offset=12)
        if count and (indices[-1] >= width or np.any(indices[1:] <= indices[:-1])):
            raise ValueError("invalid sparse observation positions")
        if count:
            positions.append(indices.astype(np.int64) + row * width)
            values.append(np.frombuffer(sample.content, dtype="<i4", count=count, offset=12 + 4 * count))
    if positions:
        indices = torch.from_numpy(np.concatenate(positions)).to(device)
        words = torch.from_numpy(np.concatenate(values)).to(device)
        target.view(torch.int32).reshape(-1)[indices] = words
    return target


class SparseTransferRolloutBuffer(PackedRolloutBuffer):
    def _get_samples(self, batch_inds, *normalizers):
        # RolloutBuffer does not apply its optional VecNormalize argument here.
        observations = restore_batch(self.observations.samples[batch_inds], self.device)
        data = (self.actions[batch_inds].astype(np.float32, copy=False),
                self.values[batch_inds].flatten(), self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(), self.returns[batch_inds].flatten())
        return RolloutBufferSamples(observations, *tuple(map(self.to_torch, data)))

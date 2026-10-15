"""Store padded numeric observations without losing any array bits."""
from dataclasses import dataclass
import zlib

import numpy as np

SPARSE_WORDS = b"SRLSP01\0"


@dataclass(frozen=True)
class PackedObservation:
    shape: tuple
    dtype: str
    content: bytes

    @classmethod
    def pack(cls, observation):
        values = np.asarray(observation)
        if values.dtype.hasobject:
            raise TypeError("observation storage requires a numeric array")
        if values.dtype.itemsize == 4 and values.size < 2**32:
            words = np.ascontiguousarray(values).reshape(-1).view("<u4")
            # Boolean scans are faster for padded observations. Count first so
            # dense arrays can go straight to zlib without allocating indices.
            present = words != 0
            count = np.count_nonzero(present)
            if 12 + 8 * count < values.nbytes:
                indices = np.flatnonzero(present)
                positions = indices.astype("<u4")
                content = (SPARSE_WORDS + len(indices).to_bytes(4, "little")
                           + positions.tobytes() + words[indices].tobytes())
                return cls(values.shape, values.dtype.str, content)
        return cls(values.shape, values.dtype.str, zlib.compress(values.tobytes(), 1))

    def unpack(self):
        result = np.empty(self.shape, dtype=self.dtype)
        self.restore_into(result)
        return result

    def restore_into(self, target):
        if target.shape != self.shape or target.dtype != np.dtype(self.dtype) or not target.flags.c_contiguous:
            raise ValueError("observation restore target must match the contiguous stored layout")
        if self.content.startswith(SPARSE_WORDS):
            count = int.from_bytes(self.content[8:12], "little")
            if target.dtype.itemsize != 4 or len(self.content) != 12 + 8 * count or count > target.size:
                raise ValueError("invalid sparse observation header")
            indices = np.frombuffer(self.content, dtype="<u4", count=count, offset=12)
            if count and (indices[-1] >= target.size or np.any(indices[1:] <= indices[:-1])):
                raise ValueError("invalid sparse observation positions")
            words = target.reshape(-1).view("<u4")
            words.fill(0)
            words[indices] = np.frombuffer(self.content, dtype="<u4", count=count, offset=12 + 4 * count)
        else:
            # Old checkpoints remain readable. Raw words preserve signed zero
            # and NaN payloads; dense and image arrays retain the zlib codec.
            values = np.frombuffer(zlib.decompress(self.content), dtype=self.dtype).reshape(self.shape)
            np.copyto(target, values)


class PackedArray:
    """Materialize only the sampled observations; retain upstream buffer indexing."""
    def __init__(self, samples, observation_shape, dtype):
        self.samples, self.observation_shape, self.dtype = samples, tuple(observation_shape), np.dtype(dtype)

    @property
    def shape(self):
        return (*self.samples.shape, *self.observation_shape)

    def __setitem__(self, key, values):
        shape = np.shape(self.samples[key])
        if values.shape != (*shape, *self.observation_shape) or values.dtype != self.dtype:
            raise ValueError("rollout observation shape or dtype changed")
        packed = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            packed[index] = PackedObservation.pack(values[index])
        self.samples[key] = packed

    def __getitem__(self, key):
        selected = np.asarray(self.samples[key], dtype=object)
        result = np.empty((*selected.shape, *self.observation_shape), dtype=self.dtype)
        for index in np.ndindex(selected.shape):
            selected[index].restore_into(result[index])
        return result

    def swapaxes(self, first, second):
        return PackedArray(self.samples.swapaxes(first, second), self.observation_shape, self.dtype)

    def reshape(self, *shape):
        if tuple(shape[-len(self.observation_shape):]) != self.observation_shape:
            raise ValueError("buffer flattening must preserve each observation")
        return PackedArray(self.samples.reshape(shape[:-len(self.observation_shape)]),
                           self.observation_shape, self.dtype)

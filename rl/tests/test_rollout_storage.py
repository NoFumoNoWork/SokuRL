"""Lossless observation storage must leave upstream PPO minibatches unchanged."""
import numpy as np
import pytest
import zlib

from soku_rl.rl.storage import PackedArray, PackedObservation, SPARSE_WORDS


def test_storage_preserves_every_bit_and_environment_major_order():
    bits = np.array([0, 0x80000000, 0x7FC01234, 0xFFFFFFFF, 0x3F800000, 1], dtype=np.uint32)
    values = np.tile(bits.view(np.float32), (3, 2, 1))
    values[1, 1, -1] = 23.
    packed = PackedArray(np.empty((3, 2), dtype=object), (6,), np.float32)
    for step in range(3):
        packed[step] = values[step]
    flattened = packed.swapaxes(0, 1).reshape(6, 6)
    order = np.array([5, 1, 3, 0, 4, 2])
    assert flattened[order].tobytes() == values.swapaxes(0, 1).reshape(6, 6)[order].tobytes()
    zeros = np.zeros(600_000, np.float32)
    saved = PackedObservation.pack(zeros)
    assert saved.unpack().tobytes() == zeros.tobytes()
    assert len(saved.content) < zeros.nbytes // 100


def test_packed_buffer_matches_upstream_returns_and_samples():
    torch = pytest.importorskip("torch")
    from gymnasium import spaces
    from stable_baselines3.common.buffers import RolloutBuffer
    from soku_rl.rl.buffers import PackedRolloutBuffer
    from soku_rl.rl.sparse_transfer import SparseTransferRolloutBuffer
    arguments = (4, spaces.Box(-100, 100, (6,), np.float32), spaces.Discrete(3))
    buffers = [kind(*arguments, device="cpu", gamma=1., gae_lambda=.95, n_envs=2)
               for kind in (RolloutBuffer, PackedRolloutBuffer, SparseTransferRolloutBuffer)]
    for step in range(4):
        obs = np.arange(12, dtype=np.float32).reshape(2, 6) + step
        for buffer in buffers:
            buffer.add(obs, np.array([1, 2]), np.array([.2, -.2]), np.array([step == 0, step == 2]),
                       torch.tensor([.3, -.4]), torch.tensor([-.1, -.2]))
    outputs = []
    for buffer in buffers:
        buffer.compute_returns_and_advantage(torch.tensor([.2, -.3]), np.array([False, True]))
        np.random.seed(32)
        outputs.append(list(buffer.get(3)))
    for candidate in outputs[1:]:
        for first, second in zip(outputs[0], candidate, strict=True):
            assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_device_restore_preserves_sparse_dense_and_legacy_words():
    torch = pytest.importorskip("torch")
    from soku_rl.rl.sparse_transfer import restore_batch
    sparse = np.zeros(200, np.float32)
    sparse.view(np.uint32)[[1, 123, 199]] = [0x80000000, 0x7FC01234, 0xFFFFFFFF]
    dense = np.arange(200, dtype=np.float32) + 1
    samples = [PackedObservation.pack(sparse), PackedObservation.pack(dense),
               PackedObservation(sparse.shape, sparse.dtype.str, zlib.compress(sparse.tobytes())),
               PackedObservation.pack(np.zeros_like(sparse))]
    result = restore_batch(samples, torch.device("cpu"))
    assert result.numpy().tobytes() == np.stack([s.unpack() for s in samples]).tobytes()
    with pytest.raises(ValueError, match="nonempty"):
        restore_batch([], torch.device("cpu"))
    malformed = PackedObservation(sparse.shape, sparse.dtype.str, SPARSE_WORDS + b"\xff" * 4)
    with pytest.raises(ValueError, match="header"):
        restore_batch([malformed], torch.device("cpu"))


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<u4"])
def test_sparse_words_preserve_nan_payloads_signed_zero_and_byte_order(dtype):
    values = np.zeros((5, 200), dtype=dtype)
    words = values.view("<u4")
    words.flat[[1, 123, 654, 999]] = [0x80000000, 0x7FC01234, 0xFFFFFFFF, 1]
    packed = PackedObservation.pack(values)
    assert packed.content.startswith(SPARSE_WORDS)
    assert packed.unpack().tobytes() == values.tobytes()
    sliced = values[:, ::2]
    assert PackedObservation.pack(sliced).unpack().tobytes() == sliced.tobytes()
    old = PackedObservation(values.shape, values.dtype.str, zlib.compress(values.tobytes(), 1))
    assert old.unpack().tobytes() == values.tobytes()


def test_dense_and_image_observations_keep_legacy_codec():
    for values in (np.arange(300, dtype=np.float32) + 1, np.full((3, 24, 32), 255, np.uint8)):
        packed = PackedObservation.pack(values)
        assert not packed.content.startswith(SPARSE_WORDS)
        assert packed.unpack().tobytes() == values.tobytes()

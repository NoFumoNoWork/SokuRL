import numpy as np
import pytest
import torch
from gymnasium import spaces

from soku_rl.rl.token_features import PublicTokenFeatures


def example():
    x = torch.zeros(2, 400 * 4 + 64)
    frames = x[:, :1600].reshape(2, 4, 400)
    frames[..., :16] = .5
    frames[..., 16:25] = torch.tensor([1., .1, .2, 1., .3, .4, 1., .5, .6])
    return x


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_mask_permutation_gradients_and_roundtrip(device, tmp_path):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(4)
    torch.manual_seed(1732)
    space = spaces.Box(-1, 1, (1664,), np.float32)
    model = PublicTokenFeatures(space).to(device)
    x = example().to(device)
    reference = model(x)
    permuted = x.clone()
    objects = permuted[:, :1600].reshape(2, 4, 400)[...,16:208].reshape(2,4,64,3)
    objects[...,:3,:] = objects[...,[2,0,1],:].clone()
    torch.testing.assert_close(model(permuted), reference, atol=1e-6, rtol=1e-5)
    noisy = x.clone()
    masked = noisy[:, :1600].reshape(2,4,400)[...,16:].reshape(2,4,128,3)
    masked[...,3:,1:] = 100.
    torch.testing.assert_close(model(noisy), reference, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(model(torch.zeros_like(x))).all()
    before = model.fighter[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    reference.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.fighter[0].weight)
    file = tmp_path / 'features.pt'
    torch.save(model.state_dict(), file)
    restored = PublicTokenFeatures(space).to(device)
    restored.load_state_dict(torch.load(file, map_location=device, weights_only=True))
    torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        PublicTokenFeatures(spaces.Box(-1,1,(400,),np.float32))
    with pytest.raises(ValueError):
        PublicTokenFeatures(spaces.Box(-1,1,(1600,),np.float32), width=63)


def test_causal_history_and_batch_isolation():
    torch.set_num_threads(4)
    torch.manual_seed(1732)
    model = PublicTokenFeatures(spaces.Box(-1,1,(1664,),np.float32))
    sequence = torch.randn(2,4,64)
    changed = sequence.clone()
    changed[:,3] += 10
    before = model.temporal(sequence,mask=model.causal)
    after = model.temporal(changed,mask=model.causal)
    torch.testing.assert_close(before[:,:3],after[:,:3],atol=0,rtol=0)
    x = example()
    baseline = model(x)
    x[1,:16] += .2
    torch.testing.assert_close(model(x)[0],baseline[0],atol=0,rtol=0)

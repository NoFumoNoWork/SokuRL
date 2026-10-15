from dataclasses import replace
from types import SimpleNamespace

from gymnasium import spaces
import numpy as np
import pytest
import torch

from bridge_shared import RawFrameState
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.observation.animation import (observe_animation, frame_width,
    HISTORY_OFFSETS, AnimationObservation, public_history)
from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.env.observation.render_state import RenderEntity, RenderSnapshot
from soku_rl.env.observation_history import ObservationHistory
from soku_rl.rl.animation_features import AnimationTokenFeatures


def example():
    visibility = VisibilityConfig(8, .5, .02, .1, 48, 96, 16, .25)
    raw = RawFrameState()
    for index, p in enumerate((raw.p1, raw.p2)):
        p.hp, p.spirit, p.maxSpirit = 10000, 1000, 1000
        p.characterId, p.actionId, p.sequenceId = index, 400+index, 2
        p.subsequenceId, p.animationFrame, p.elapsedInSubsequence = 3, 4, 5
    raw.p1ObjectCount = 1
    obj = raw.p1Objects[0]
    obj.x, obj.y, obj.actionId, obj.actionBlockId = 300, 100, 800, 1
    render = RenderSnapshot(0, 400, 1, 0,
        (RenderEntity(100, 0, 1, 1, 1), RenderEntity(500, 0, 1, -1, 1)),
        ((RenderEntity(300, 100, 1, 1, 1),), ()), False)
    return raw, render, visibility


def test_visible_identity_and_no_hidden_categories():
    raw, render, v = example()
    first, second = observe_animation(raw, render, v)
    assert len(first.values) == frame_width(16)
    assert first.values[5] == 400 and second.values[5] == 401
    assert first.values[25] == 800 and first.values[29] == 0
    assert second.values[29] == 1
    hidden = replace(render, objects=((replace(render.objects[0][0], alpha=0),), ()))
    assert not any(observe_animation(raw, hidden, v)[0].values[20:-5])
    obj = raw.p1Objects[0]
    obj.typeId, obj.collisionType, obj.speedX = 123456, 9, 999
    assert observe_animation(raw, render, v)[0] == first
    raw.p1ObjectOverflow = 1
    with pytest.raises(RuntimeError, match='overflow'):
        observe_animation(raw, render, v)


def test_association_failure_is_explicit():
    raw, render, v = example()
    raw.p1Objects[0].x = 301
    with pytest.raises(RuntimeError, match='identity mismatch'):
        observe_animation(raw, render, v)


def test_sparse_history_exact_past_and_reset():
    _, _, visibility = example()
    config = EpisodeConfig(7200, 8, 1, 0, 'animation', visibility, LEGACY_MATCH)
    history = ObservationHistory(config, ('player_0', 'player_1'))
    def pair(frame):
        obs = AnimationObservation(frame, tuple([float(frame)] * frame_width(16)))
        return obs, obs
    history.reset(0, pair(0))
    for frame in range(1, 14):
        history.append(frame, pair(frame))
    values = history.observations()['player_0'].reshape(8, -1)
    assert values[:, 0].tolist() == [13-offset for offset in HISTORY_OFFSETS]
    history.reset(0, pair(0))
    assert not history.observations()['player_0'].any()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_animation_network_mask_gradients_and_categories(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(4)
    raw, render, v = example()
    values = observe_animation(raw, render, v)[0].values
    x = torch.tensor([list(values)*8+[0.]*64], device=device)
    model = AnimationTokenFeatures(spaces.Box(-np.inf, np.inf, (x.shape[1],), np.float32)).to(device)
    y = model(x)
    assert y.shape == (1, 128) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert model.categories[1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='category'):
        bad = x.clone()
        bad[0, 5] = 999999
        model(bad)
    hidden = x.clone()
    hidden[:, :1480].reshape(1, 8, -1)[..., 30:34] = torch.tensor([0, .9, .9, 1], device=device)
    torch.testing.assert_close(model(hidden), y, atol=1e-6, rtol=1e-5)


def test_public_adapter_is_lossy_and_correct():
    raw, render, v = example()
    values = observe_animation(raw, render, v)[0].values
    public = public_history(np.array(values*8), 16).reshape(8, 400)
    assert public[0, 5] == 1 and public[0, 13] == 1
    assert public[0, 16] == 1

"""Reproduce the signed spirit word recorded during a real guard break."""
from types import SimpleNamespace

import pytest

from soku_rl.render_state import RenderEntity, RenderSnapshot
from soku_rl.visible_state import observe_visible_states
from test_env_timing import VISIBILITY


def recorded_frame(spirit):
    fighters = [SimpleNamespace(characterId=1, hp=2201, spirit=876, maxSpirit=1000),
                SimpleNamespace(characterId=0, hp=4783, spirit=spirit, maxSpirit=1000)]
    raw = SimpleNamespace(frameId=3185, p1=fighters[0], p2=fighters[1])
    render = RenderSnapshot(-700., 470., 1., 21,
        (RenderEntity(1201., 312.70535, 1., 1, 1),
         RenderEntity(1240., 370.81131, 1., -1, 1)), ((), ()), False)
    return raw, render


def test_guard_break_negative_spirit_is_an_empty_visible_gauge():
    raw, render = recorded_frame(65448)  # signed 16-bit value -88, exported as uint32
    first, second = observe_visible_states(raw, render, VISIBILITY)
    assert first.frame == second.frame == 3185
    assert first.values[14] == second.values[6] == 0.
    assert first.values[6] == second.values[14] == .9


@pytest.mark.parametrize("invalid", [-88, 65536, 1001])
def test_malformed_spirit_words_and_positive_overflow_still_fail(invalid):
    raw, render = recorded_frame(invalid)
    with pytest.raises(ValueError, match="spirit|gauge"):
        observe_visible_states(raw, render, VISIBILITY)

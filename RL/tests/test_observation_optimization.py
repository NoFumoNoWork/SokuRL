from types import SimpleNamespace
import random

import pytest

from game_runtime.observation import ObservationReader
from soku_rl.env.observation import visible_state
from soku_rl.env.observation.visibility import ScreenEntity, VisibilityConfig


CONFIG = VisibilityConfig(8, .5, .02, .1, 48., 96., 16., .25)


def test_shared_encoding_equals_independent_seats(monkeypatch):
    rng = random.Random(1732)
    for frame in range(200):
        fighters = [SimpleNamespace(hp=rng.randint(0, 10000), spirit=rng.randint(0, 1000),
                    maxSpirit=1000, characterId=rng.randrange(20)) for _ in range(2)]
        raw = SimpleNamespace(frameId=frame, p1=fighters[0], p2=fighters[1])
        poses = tuple(ScreenEntity(bool(rng.randrange(2)), rng.randrange(80)*8.,
                      rng.randrange(60)*8., rng.choice((-1, 1))) for _ in range(2))
        objects = tuple(tuple(ScreenEntity(True, rng.randrange(80)*8.,
                       rng.randrange(60)*8., 1) for _ in range(rng.randrange(75))) for _ in range(2))
        monkeypatch.setattr(visible_state, "visible_entities", lambda render, config: (poses, objects))
        expected = tuple(visible_state._encode(raw, poses, objects, p, CONFIG) for p in (0, 1))
        assert visible_state.observe_visible_states(raw, None, CONFIG) == expected


def test_known_dropped_reuses_validated_snapshot_and_fallback_still_checks():
    reader = ObservationReader.__new__(ObservationReader)
    reader.mode, reader.pid = "diagnostic_state", 123
    reader.diagnostic = SimpleNamespace(observe=lambda raw, bridge: ("a", "b"))
    raw = SimpleNamespace(sceneId=5, battleMode=3, p1=SimpleNamespace(hp=10000, characterId=0),
        p2=SimpleNamespace(hp=10000, characterId=1), frameId=10, segmentId=0, stageId=0,
        activeWeather=21, stateHash=0, p1ObjectCount=0, p2ObjectCount=0)
    calls = []
    bridge = SimpleNamespace(snapshot=lambda: calls.append(True) or SimpleNamespace(dropped_frames=1))
    reader.read(raw, bridge, known_dropped=0)
    assert calls == []
    with pytest.raises(RuntimeError, match="incomplete"):
        reader.read(raw, bridge)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="incomplete"):
        reader.read(raw, bridge, known_dropped=1)

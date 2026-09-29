from dataclasses import replace
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

from soku_rl.baselines import Fighter, Observation, Projectile
from soku_rl.env import EpisodeConfig
from soku_rl.env.encoding import decode_action, encode_observation
from soku_rl.observed_rules import RulePolicy, decode_diagnostic


def episode(mode):
    raw = OmegaConf.to_container(OmegaConf.load(Path(__file__).parents[1] / "config/train.yaml"))
    return EpisodeConfig(**(raw["episode"] | {"decision_frames": 3 if mode == "state" else 1,
        "latency_frames": 12 if mode == "state" else 0, "observation_mode": mode}))


def policy(name, mode):
    rules = OmegaConf.to_container(OmegaConf.load(Path(__file__).parents[1] / "config/rules/default.yaml"))
    return RulePolicy(name, rules, episode(mode), "test")


def public_frame():
    frame = np.zeros(400, dtype=np.float32)
    frame[:8] = (1, .3, .8, 1, 1, 1, 1, 1 / 19)
    frame[8:16] = (1, .52, .8, -1, 1, 1, 1, 0)
    return frame


def test_diagnostic_roundtrip():
    own = Fighter(411, 0, 8350, .6, 321, False, 3, 1, 1)
    enemy = Fighter(601, 88, 9900, .8, 50, True, 2, 0, -1)
    original = Observation(601, own, enemy, (Projectile(101, 90, -12, 3),))
    decoded = decode_diagnostic(encode_observation(original, 7200), 7200)
    assert decoded.frame == original.frame
    assert decoded.player.action_id == 321
    assert decoded.opponent.airborne
    assert decoded.player.hp == 8350
    assert decoded.enemy_projectiles[0].speed_x == pytest.approx(-12)


def test_hidden_pose_never_drives_direction_or_special():
    frame = public_frame()
    frame[8:12] = (0, 0, 0, 0)
    for name in ("rush", "zoning", "counter", "community_combo", "community_guard"):
        assert policy(name, "state").spawn(5).act(np.tile(frame, 4)) == 256


def test_screen_special_duration_and_episode_isolation():
    spec = policy("community_combo", "state")
    obs = np.tile(public_frame(), 4)
    actor = spec.spawn(5)
    actions = [decode_action(actor.act(obs)).inputs for _ in range(4)]
    assert [a[:2] for a in actions] == [(0, 1), (1, 1), (1, 0), (0, 0)]
    assert actions[2][3] == 1
    assert spec.spawn(5).act(obs) == spec.spawn(6).act(obs)


def test_numeric_shape_rejected_and_identity_contains_timing():
    spec = policy("rush", "state")
    with pytest.raises(ValueError, match="configuration"):
        spec.spawn(1).act(np.zeros(1356))
    assert spec.fingerprint != replace(spec, episode=replace(spec.episode, latency_frames=9)).fingerprint


def test_diagnostic_community_keeps_frame_contract():
    spec = policy("community_combo", "diagnostic_state")
    own = Fighter(400, 0, 10000, 1, 0, False, 0, 1, 1)
    enemy = Fighter(700, 0, 10000, 1, 0, False, 0, 0, -1)
    actor = spec.spawn(1)
    for frame, axes in enumerate(((0, 1), (1, 1), (1, 0))):
        obs = encode_observation(Observation(frame, own, enemy, ()), 7200)
        assert decode_action(actor.act(np.tile(obs, 4))).inputs[:2] == axes

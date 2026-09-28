from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from soku_rl.baselines import Fighter, Observation
from soku_rl.strategies import strategy_from_config


def config():
    return OmegaConf.to_container(OmegaConf.load(Path(__file__).parents[1] / "config/baselines.yaml"))


def strategy(name):
    return strategy_from_config(name, config(), "test-version")


def observation():
    own = Fighter(400, 0, 10000, 1.0, 0, False, 0, 1, 1)
    enemy = Fighter(700, 0, 10000, 1.0, 0, False, 0, 0, -1)
    return Observation(0, own, enemy, ())


def test_236_sequence_and_mirror():
    for facing in (-1, 1):
        obs = observation()
        obs = replace(obs, player=replace(obs.player, facing=facing),
                      opponent=replace(obs.opponent, x=400 + facing * 300))
        policy = strategy("community_combo").spawn(1)
        inputs = [policy.act(replace(obs, frame=f)).inputs for f in range(6)]
        assert [a[:2] for a in inputs] == [(0, 1), (facing, 1), (facing, 0), (facing, 0), (0, 0), (0, 0)]
        assert [a[3] for a in inputs] == [0, 0, 1, 1, 0, 0]


def test_hit_confirm_in_hitstop_and_release():
    obs = observation()
    obs = replace(obs, player=replace(obs.player, action_id=300, hitstop=3),
                  opponent=replace(obs.opponent, action_id=50, x=450))
    policy = strategy("community_combo").spawn(1)
    assert policy.act(obs).rule == "confirmed_A_chain"
    assert policy.act(replace(obs, frame=1)).inputs[2] == 0
    assert policy.act(replace(obs, frame=2)).inputs[2] == 1


def test_guard_distinguishes_high_and_low():
    for action, vertical in [(303, 1), (300, 0)]:
        obs = observation()
        obs = replace(obs, opponent=replace(obs.opponent, x=450, action_id=action))
        assert strategy("community_guard").spawn(1).act(obs).inputs[:2] == (-1, vertical)


def test_damage_interrupts_pending_command():
    policy = strategy("community_combo").spawn(1)
    obs = observation()
    assert policy.act(obs).rule == "236_down"
    damaged = replace(obs, frame=1, player=replace(obs.player, action_id=50))
    assert policy.act(damaged).rule == "wait_hitstun"
    assert not policy.pending
    assert policy.act(replace(obs, frame=2)).rule != "236_down_forward"


def test_episode_isolation_and_invalid_observations():
    spec = strategy("community_combo")
    first, second = spec.spawn(1), spec.spawn(1)
    obs = observation()
    first.act(obs)
    assert second.act(obs).rule == "236_down"
    with pytest.raises(ValueError, match="consecutive"):
        first.act(obs)
    with pytest.raises(ValueError, match="only Reimu"):
        spec.spawn(1).act(replace(obs, player=replace(obs.player, character_id=2)))


def test_strategy_fingerprint_tracks_configuration():
    cfg = config()
    first = strategy_from_config("community_combo", cfg, "v1")
    cfg["community"]["community_combo"]["special_cooldown"] = 180
    second = strategy_from_config("community_combo", cfg, "v1")
    assert first.fingerprint != second.fingerprint
    assert first.fingerprint == strategy_from_config("community_combo", config(), "v1").fingerprint

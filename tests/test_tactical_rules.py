"""Check tactical behavior and the public, diagnostic and learner contracts."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

from soku_rl.baselines import Fighter, Observation, Projectile
from soku_rl.env import EpisodeConfig
from soku_rl.env.encoding import decode_action, encode_action, encode_observation
from soku_rl.learning_wrappers import LearningConfig, LearningInterface, LearningRulePolicy
from soku_rl.observed_rules import RulePolicy
from soku_rl.strategies import rule_implementation, strategy_from_config
from soku_rl.tactical_rules import TACTICAL_STYLES, TacticalConfig


ROOT = Path(__file__).parents[1]
RULES = OmegaConf.to_container(OmegaConf.load(ROOT / "config/rules/default.yaml"))
BASE = Observation(0, Fighter(400, 0, 10000, 1., 0, False, 0, 1, 1),
                   Fighter(720, 0, 10000, 1., 0, False, 0, 0, -1), ())


def actor(name):
    return strategy_from_config(name, RULES, "test").spawn(17)


def episode(mode):
    raw = OmegaConf.to_container(OmegaConf.load(ROOT / "config/train.yaml"))
    return EpisodeConfig(**(raw["episode"] | {"observation_mode": mode,
        "decision_frames": 3 if mode == "state" else 1,
        "latency_frames": 12 if mode == "state" else 0}))


def tensor(observation, mode):
    if mode == "diagnostic_state":
        return np.tile(encode_observation(observation, 7200), 4)
    frame = np.zeros(400, dtype=np.float32)
    for index, fighter in enumerate((observation.player, observation.opponent)):
        frame[index * 8:index * 8 + 8] = (1, fighter.x / 1280, .8 - fighter.y / 960,
            fighter.facing, 1, fighter.hp / 10000, fighter.spirit_fraction, fighter.character_id / 19)
    for index, projectile in enumerate(observation.enemy_projectiles):
        frame[208 + index * 3:211 + index * 3] = (1, projectile.x / 1280, .8 - projectile.y / 960)
    return np.tile(frame, 4)


def mirror(observation):
    def fighter(value):
        return replace(value, x=1280 - value.x, facing=-value.facing)
    return replace(observation, player=fighter(observation.player), opponent=fighter(observation.opponent),
        enemy_projectiles=tuple(replace(p, x=1280 - p.x, speed_x=-p.speed_x)
                                for p in observation.enemy_projectiles))


@pytest.mark.parametrize("mode", ("state", "diagnostic_state"))
@pytest.mark.parametrize("name", TACTICAL_STYLES)
def test_episode_isolation_mirroring_and_attack_release(name, mode):
    spec = RulePolicy(name, RULES, episode(mode), "test")
    first, second, reflected = spec.spawn(3), spec.spawn(4), spec.spawn(5)
    stride = spec.episode.decision_frames
    previous = (0,) * 8
    for frame in range(0, 192, stride):
        obs = replace(BASE, frame=frame)
        action = decode_action(first.act(tensor(obs, mode))).inputs
        assert action == decode_action(second.act(tensor(obs, mode))).inputs
        reversed_action = decode_action(reflected.act(tensor(mirror(obs), mode))).inputs
        assert reversed_action == (-action[0], *action[1:])
        assert not any(a and b for a, b in zip(action[2:5], previous[2:5]))
        previous = action
    assert spec.spawn(3).act(tensor(BASE, mode)) == spec.spawn(9).act(tensor(BASE, mode))


@pytest.mark.parametrize("mode", ("state", "diagnostic_state"))
def test_ten_tactics_have_distinct_actions_even_with_identical_distances(mode):
    rules = deepcopy(RULES)
    for name in TACTICAL_STYLES:
        rules["tactical"][name]["movement"] = "counter"
    scenarios = [
        BASE,
        replace(BASE, opponent=replace(BASE.opponent, x=455)),
        replace(BASE, opponent=replace(BASE.opponent, x=540, y=180)),
        replace(BASE, player=replace(BASE.player, y=180)),
        replace(BASE, player=replace(BASE.player, x=1100), opponent=replace(BASE.opponent, x=1230)),
        replace(BASE, opponent=replace(BASE.opponent, x=620, spirit_fraction=.1)),
        replace(BASE, enemy_projectiles=(Projectile(440, 50, -10, 0),)),
    ]
    traces = {}
    for name in TACTICAL_STYLES:
        trace = []
        for scenario in scenarios:
            spec = RulePolicy(name, rules, episode(mode), "test")
            current = spec.spawn(1)
            for frame in range(0, 96, spec.episode.decision_frames):
                obs = replace(scenario, frame=frame)
                trace.append(current.act(tensor(obs, mode)))
        traces[name] = tuple(trace)
    assert len(set(traces.values())) == 10


def test_pressure_mixes_low_and_forward_attacks():
    current = actor("pressure")
    obs = replace(BASE, opponent=replace(BASE.opponent, x=450))
    attacks = [current.act(replace(obs, frame=frame)).inputs for frame in range(17)]
    assert [a[:2] for a in attacks if a[2]] == [(0, 0), (0, 1), (1, 0)]


def test_footsies_retreats_then_punishes_when_approach_stops():
    current = actor("footsies")
    current.act(BASE)
    near = replace(BASE, frame=1, opponent=replace(BASE.opponent, x=520))
    assert current.act(near).inputs[:2] == (-1, 0)
    assert current.act(replace(near, frame=2)).rule == "spacing_punish"


@pytest.mark.parametrize("character,horizontal", ((0, 0), (1, -1)))
def test_anti_air_uses_character_specific_close_melee(character, horizontal):
    obs = replace(BASE, player=replace(BASE.player, character_id=character),
                  opponent=replace(BASE.opponent, x=450, y=150))
    assert actor("anti_air").act(obs).inputs == (horizontal, 0, 1, 0, 0, 0, 0, 0)


def test_air_rush_changes_from_jump_to_flight_to_shot():
    current = actor("air_rush")
    actions = [current.act(replace(BASE, frame=f)) for f in range(37)]
    assert actions[0].inputs[:2] == (1, -1)
    assert actions[18].inputs[5] == 1
    assert actions[36].inputs[3] == 1


def test_bullet_wall_layers_shots_with_jump_cancels():
    current = actor("bullet_wall")
    actions = [current.act(replace(BASE, frame=f)) for f in range(49)]
    assert actions[0].inputs[3] == 1
    assert actions[1].inputs[:2] == (-1, -1)
    assert actions[24].inputs[4] == 1
    assert actions[48].inputs[1:5] == (1, 0, 0, 1)


def test_graze_hunter_retains_chase_after_projectile_passes():
    current = actor("graze_hunter")
    obs = replace(BASE, enemy_projectiles=(Projectile(440, 50, -10, 0),))
    assert current.act(obs).inputs == (1, 0, 0, 0, 0, 1, 0, 0)
    assert current.act(replace(BASE, frame=1)).rule == "continue_graze_chase"
    close = replace(BASE, frame=2, opponent=replace(BASE.opponent, x=520))
    assert current.act(close).rule == "graze_dash_attack"


def test_hit_and_run_withdraws_for_a_bounded_duration():
    current = actor("hit_and_run")
    obs = replace(BASE, opponent=replace(BASE.opponent, x=520))
    actions = [current.act(replace(obs, frame=f)) for f in range(37)]
    assert actions[0].inputs[2] == actions[36].inputs[2] == 1
    assert all(a.inputs == (-1, 0, 0, 0, 0, 0, 0, 0) for a in actions[1:36])


def test_corner_trap_covers_jump_without_chasing_over_opponent():
    obs = replace(BASE, player=replace(BASE.player, x=1100),
                  opponent=replace(BASE.opponent, x=1230, y=150))
    assert actor("corner_trap").act(obs).inputs == (0, 1, 0, 0, 1, 0, 0, 0)


def test_spirit_siege_changes_shot_when_enemy_resource_is_low():
    assert actor("spirit_siege").act(BASE).inputs[3] == 1
    obs = replace(BASE, opponent=replace(BASE.opponent, spirit_fraction=.1))
    assert actor("spirit_siege").act(obs).inputs[4] == 1


@pytest.mark.parametrize("mode", ("state", "diagnostic_state"))
def test_skill_sequence_uses_exactly_one_command_per_decision(mode):
    spec = RulePolicy("skill_cycle", RULES, episode(mode), "test")
    current = spec.spawn(1)
    actions = [decode_action(current.act(tensor(replace(BASE, frame=f), mode))).inputs
               for f in range(0, 4 * spec.episode.decision_frames, spec.episode.decision_frames)]
    assert [a[:2] for a in actions] == [(0, 1), (1, 1), (1, 0), (0, 0)]
    assert [a[3] for a in actions] == [0, 0, 1, 0]


@pytest.mark.parametrize("mode", ("state", "diagnostic_state"))
def test_damage_and_crossing_cancel_skill_sequence(mode):
    spec = RulePolicy("skill_cycle", RULES, episode(mode), "test")
    for changed in (replace(BASE, player=replace(BASE.player, hp=8000)),
                    replace(BASE, opponent=replace(BASE.opponent, x=100))):
        current = spec.spawn(1)
        current.act(tensor(BASE, mode))
        obs = replace(changed, frame=spec.episode.decision_frames)
        action = decode_action(current.act(tensor(obs, mode))).inputs
        assert action != (1, 1, 0, 0, 0, 0, 0, 0)
        tactical = current.actor.actor if mode == "state" else current.actor
        assert not tactical.pending


def test_invisible_pose_clears_commands_without_using_hidden_positions():
    spec = RulePolicy("skill_cycle", RULES, episode("state"), "test")
    current = spec.spawn(1)
    current.act(tensor(BASE, "state"))
    hidden = tensor(BASE, "state").reshape(4, 400)
    hidden[-1, 8:12] = (0, .95, .1, 0)
    assert current.act(hidden.ravel()) == 256
    assert not current.actor.actor.pending
    assert not current.actor.actor.history
    assert decode_action(current.act(tensor(BASE, "state"))).inputs[3] == 0


@pytest.mark.parametrize("name", TACTICAL_STYLES)
def test_resource_recovery_has_hysteresis_and_terminal_is_neutral(name):
    current = actor(name)
    for frame, spirit in enumerate((.1, .3, .5)):
        obs = replace(BASE, frame=frame, player=replace(BASE.player, spirit_fraction=spirit))
        assert current.act(obs).rule == "recover_spirit"
    assert current.act(replace(BASE, frame=3)).rule != "recover_spirit"
    dead = replace(BASE, frame=4, opponent=replace(BASE.opponent, hp=0))
    assert encode_action(current.act(dead).inputs) == 256


@pytest.mark.parametrize("name", TACTICAL_STYLES)
@pytest.mark.parametrize("mode", ("state", "diagnostic_state"))
def test_learning_wrapper_preserves_tactical_actions(name, mode):
    spec = RulePolicy(name, RULES, episode(mode), "test")
    interface = LearningInterface(spec.episode, LearningConfig("combat", False, 0, 0.))
    wrapped, raw = LearningRulePolicy(spec, interface).spawn(1), spec.spawn(1)
    for frame in range(0, 96, spec.episode.decision_frames):
        values = tensor(replace(BASE, frame=frame), mode)
        assert interface.command(wrapped.act(values)) == raw.act(values)


def test_frame_and_config_errors_fail_early():
    current = actor("pressure")
    with pytest.raises(ValueError, match="consecutive"):
        current.act(replace(BASE, frame=1))
    current.act(BASE)
    with pytest.raises(ValueError, match="consecutive"):
        current.act(BASE)
    values = RULES["tactical_defaults"] | {"style": "pressure"}
    for change in ({"guard_frames": 1}, {"shot_interval": 3.5},
                   {"spirit_recover": float("nan")}, {"style": "unknown"}):
        with pytest.raises(ValueError):
            TacticalConfig(**(values | change))


@pytest.mark.parametrize("mode", ("state", "diagnostic_state"))
def test_invalid_character_seed_and_numbers_are_rejected(mode):
    spec = RulePolicy("pressure", RULES, episode(mode), "test")
    for seed in (-1, 2**32, True):
        with pytest.raises(ValueError, match="seed"):
            spec.spawn(seed)
    invalid = replace(BASE, player=replace(BASE.player, character_id=2))
    with pytest.raises(ValueError, match="Reimu"):
        spec.spawn(1).act(tensor(invalid, mode))
    values = tensor(BASE, mode)
    values[-1] = np.nan
    with pytest.raises(ValueError):
        spec.spawn(1).act(values)


def test_roster_is_shared_by_training_evaluation_and_benchmark():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
        evaluation = compose(config_name="evaluation")
        training = compose(config_name="train", overrides=["algorithm=ppo"])
        benchmark = compose(config_name="benchmark")
    roster = RULES["roster"]
    assert len(roster) == len(set(roster)) == 15
    assert set(TACTICAL_STYLES) <= set(roster)
    assert "idle" not in roster
    assert list(evaluation.profiles) == list(training.algorithm.opponents) == list(benchmark.benchmark.opponents) == roster
    assert len({strategy_from_config(name, RULES, "test").fingerprint for name in roster}) == 15
    with pytest.raises(ValueError, match="unknown strategy"):
        strategy_from_config("missing", RULES, "test")


@pytest.mark.parametrize("filename", ("tactical_rules.py", "tactical_observation.py"))
def test_implementation_fingerprint_includes_new_modules(monkeypatch, filename):
    original = Path.read_bytes
    before = rule_implementation()

    def changed(path):
        data = original(path)
        return data + b"\n# changed\n" if path.name == filename else data

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert rule_implementation() != before

"""CPU ownership selection must be explicit, not a neutral scripted opponent."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from soku_rl.rl.native_cpu import NativeCPUPlaceholder, validate_configuration
from native_cpu_worker import cpu_launcher
from types import SimpleNamespace
from game_runtime.observation import time_step


def configuration():
    with initialize_config_dir(version_base='1.3', config_dir=str(Path(__file__).resolve().parents[1]/'config')):
        return OmegaConf.to_container(compose(config_name='train_combat_kickstarting_lunatic'), resolve=True)


def test_explicit_cpu_contract():
    config = configuration()
    validate_configuration(config)
    assert config['rl']['kickstarting']['teacher']['script'] == '01_marisa_main.ai'
    assert config['algorithm']['initial_policy']['kind'] == 'shared_weights'
    assert config['episode']['shared_teacher_context']


@pytest.mark.parametrize('section,key,value', [
    ('runtime','fresh_reset',True), ('runtime','record_replays',True),
    ('algorithm','player','balanced'), ('runtime','fixed_weather','typhoon'),
    ('runtime','command',['python','rollout_worker.py']),
])
def test_reject_unverified_cpu_modes(section, key, value):
    config = configuration()
    config[section][key] = value
    with pytest.raises(ValueError, match='native CPU'):
        validate_configuration(config)


def test_no_cpu_worker_with_rule_opponent():
    config = configuration()
    config['algorithm']['opponents'][0]['policy'] = {'kind':'rule'}
    with pytest.raises(ValueError):
        validate_configuration(config)


def test_existing_launcher_owns_native_cpu_flag():
    calls = []
    launch = cpu_launcher(lambda *args, **kwargs: calls.append((args,kwargs)))
    launch(2, 60, characters=(1,0), cards_disabled=True, original_replay_recording=False)
    assert calls[0][1]['native_cpu'] is True
    with pytest.raises(ValueError):
        launch(2, 60, characters=(0,1), cards_disabled=True)


def test_neutral_is_only_transport_placeholder():
    class Interface:
        def action(self, command):
            return 123
    marker = NativeCPUPlaceholder('original-lunatic',Interface())
    assert marker.spawn(1).act(None) == 123
    assert 'original-vscom' in marker.fingerprint


@pytest.mark.parametrize('mode,expected', [(3,3),(2,2)])
def test_explicit_mode_keeps_terminal_and_state_semantics(mode, expected):
    raw = SimpleNamespace(sceneId=5,battleMode=mode,frameId=0,segmentId=7,
        p1=SimpleNamespace(hp=10000,characterId=1),p2=SimpleNamespace(hp=0,characterId=0),
        stageId=0,activeWeather=0,stateHash=42,p1ObjectCount=1,p2ObjectCount=2)
    value = time_step(raw,0,('a','b'),123,expected_battle_mode=expected)
    assert value.ended
    assert value.diagnostics['hp'] == (10000,0)
    assert value.diagnostics['characters'] == (1,0)
    with pytest.raises(RuntimeError, match='incomplete'):
        time_step(raw,1,('a','b'),123,expected_battle_mode=expected)
    raw.sceneId = 3
    with pytest.raises(RuntimeError, match='left VS'):
        time_step(raw,0,('a','b'),123,expected_battle_mode=expected)


@pytest.mark.parametrize('mode,expected', [(2,3),(3,2),(1,3),(4,2)])
def test_modes_cannot_silently_fallback(mode, expected):
    raw = SimpleNamespace(sceneId=5,battleMode=mode)
    with pytest.raises(RuntimeError, match='left VS'):
        time_step(raw,0,(),0,expected_battle_mode=expected)


def test_default_rejects_cpu_mode():
    with pytest.raises(RuntimeError, match='left VS'):
        time_step(SimpleNamespace(sceneId=5,battleMode=2),0,(),0)
    with pytest.raises(ValueError):
        time_step(None,0,(),0,expected_battle_mode=True)

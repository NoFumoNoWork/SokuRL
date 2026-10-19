"""CPU ownership selection must be explicit, not a neutral scripted opponent."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from soku_rl.rl.native_cpu import NativeCPUPlaceholder, validate_configuration
from native_cpu_worker import cpu_launcher


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

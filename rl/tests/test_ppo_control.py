from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from summarize_ppo_control import paired


def test_controlled_trial_contract(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('SOKURL_RL_ROOT', root.as_posix())
    monkeypatch.setenv('SOKURL_PLATFORM_ROOT', root.parent.as_posix())
    monkeypatch.setenv('SOKURL_PYTHON', 'python')
    with initialize_config_dir(version_base='1.3', config_dir=str(root / 'config')):
        configs = [OmegaConf.to_container(compose(config_name=f'train_ppo_control_{arm}'),
                                         resolve=True) for arm in ('a', 'b', 'c')]
        matrix = OmegaConf.to_container(compose(config_name='benchmark_ppo_fixed'), resolve=True)
    a, b, c = configs
    assert all(row['algorithm']['initial_policy'] == a['algorithm']['initial_policy'] for row in configs)
    assert a['algorithm']['initial_policy']['kind'] == 'weights'
    assert all(row['algorithm']['timesteps'] == 196608 for row in configs)
    assert all(row['seed'] == a['seed'] and row['num_envs'] == 4 for row in configs)
    assert all(row['episode'] == a['episode'] and row['runtime'] == a['runtime'] for row in configs)
    assert all(row['runtime']['fresh_reset'] for row in configs)
    assert b['algorithm']['opponents'] == a['algorithm']['opponents']
    assert {k: v for k, v in a['rl']['ppo'].items() if k != 'learning_rate'} == {
        k: v for k, v in b['rl']['ppo'].items() if k != 'learning_rate'}
    assert b['rl']['ppo']['learning_rate'] == 3e-5
    assert c['rl']['ppo'] == a['rl']['ppo']
    assert [row['name'] for row in c['algorithm']['opponents']] == ['pressure']
    assert matrix['opponent_source'] == 'config'
    assert matrix['algorithm']['opponents'] == a['algorithm']['opponents']
    assert len(matrix['benchmark']['world_seeds']) == 32


def trial(seed, seat, wins, opponent_id='opponent-v1'):
    return dict(world_seed=seed, learner_seat=seat, opponent='pressure', match={},
                policy_seeds=[123, 456], strategy_ids=(['learner', opponent_id]
                if seat == 0 else [opponent_id, 'learner']),
                outcome=f'p{seat+1}_win' if wins else f'p{2-seat}_win')


def test_paired_statistics_resample_seed_blocks_and_ignore_completion_order():
    before = [trial(seed, seat, False) for seed in (1, 2) for seat in (0, 1)]
    after = [trial(seed, seat, True) for seed in (1, 2) for seat in (0, 1)]
    result = paired(before, after[::-1])['pressure']
    assert result['world_seed_blocks'] == 2
    assert result['paired_win_rate_delta'] == 1
    assert result['descriptive_block_bootstrap_95_interval'] == [1, 1]
    after[0]['strategy_ids'][1] = 'different-opponent'
    with pytest.raises(RuntimeError, match='opponent version'):
        paired(before, after)


def test_paired_statistics_reject_missing_seat():
    rows = [trial(1, 0, True)]
    with pytest.raises(RuntimeError, match='both seats'):
        paired(rows, rows)

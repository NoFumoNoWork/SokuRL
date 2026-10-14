import pytest
from evaluate_br_stage import evaluation_keys

def trial(seed,seat=0):
    return dict(world_seed=seed,learner_seat=seat,match={'player_0':{'character':1}},policy_seeds=[123,456])

def test_episode_completion_order_is_not_trial_identity():
    a = [trial(1),trial(2,1)]
    assert evaluation_keys(a)==evaluation_keys(a[::-1])
    assert evaluation_keys(a)!=evaluation_keys([trial(3),trial(2,1)])
    assert evaluation_keys(a)!=evaluation_keys([trial(1,1),trial(2,1)])

def test_duplicate_trial_is_rejected():
    with pytest.raises(RuntimeError,match='duplicate'):
        evaluation_keys([trial(1),trial(1)])

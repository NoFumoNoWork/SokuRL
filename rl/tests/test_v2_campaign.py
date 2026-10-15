from copy import deepcopy

import pytest
from summarize_v2_campaign import paired


def test_paired_comparison_does_not_require_same_learner_hash():
    reference=dict(opponent='pressure',learner_seat=0,world_seed=17,
                   policy_seeds=[11,12],match={'test':1},strategy_ids=['old','foe'],
                   outcome='p2_win')
    candidate=deepcopy(reference)
    candidate.update(outcome='p1_win',strategy_ids=['new','foe'])
    result=paired([reference],[candidate])
    assert result['gained']==1 and result['lost']==0
    candidate['policy_seeds'][0]=33
    with pytest.raises(ValueError,match='seed/match mismatch'):
        paired([reference],[candidate])


def test_opponent_version_must_be_fixed():
    reference=dict(opponent='pressure',learner_seat=1,world_seed=17,
                   policy_seeds=[11,12],match={},strategy_ids=['foe','old'],outcome='p2_win')
    candidate=deepcopy(reference);candidate['strategy_ids'][0]='new-foe'
    with pytest.raises(ValueError,match='fingerprint differs'):
        paired([reference],[candidate])

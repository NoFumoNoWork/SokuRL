from types import SimpleNamespace
import pytest

from soku_rl.env.reward import CombatReward, CombatRewardConfig
from soku_rl.pomg import Outcome


def step(frame, hp=(10000,10000), events=(), outcome=Outcome.ONGOING, **extra):
    return SimpleNamespace(frame=frame,diagnostics=dict(segment=1,hp=hp,native_combat_events=events,**extra),
        outcome=outcome,terminated=outcome in (Outcome.P1_WIN,Outcome.P2_WIN,Outcome.DRAW),
        truncated=outcome==Outcome.TRUNCATED,ended=outcome!=Outcome.ONGOING)


def hit(frame, before, after, order=1, **extra):
    return dict(frame=frame,segment=1,order=order,kind='HIT_RESOLUTION',attacker_seat=0,
        defender_seat=1,hp_before=before,hp_after=after,hit_or_block='HIT',damage_rate=.5,
        independent_input_segment=False,counter_hit_first=False,**extra)


def test_native_damage_combo_ch_bound_and_terminal_once():
    reward=CombatReward(CombatRewardConfig(passive_b=0))
    reward.reset(step(0))
    event=hit(1,10000,9000);event['counter_hit_first']=True
    values,parts=reward.step(step(1,(10000,9000),(event,)))
    assert values==pytest.approx((.12,-.1))
    event=hit(2,9000,8000);event['independent_input_segment']=True
    values,parts=reward.step(step(2,(10000,8000),(event,)))
    assert values==pytest.approx((.115,-.1))
    values,_=reward.step(step(3,(10000,8000),outcome=Outcome.P1_WIN))
    assert values==(1.,-1.)
    with pytest.raises(RuntimeError): reward.step(step(4,(10000,8000)))


def test_duplicate_unknown_and_uncovered_mutation_fail_closed():
    reward=CombatReward(CombatRewardConfig(passive_b=0))
    reward.reset(step(0))
    event=hit(1,10000,9000);event['independent_input_segment']=None
    with pytest.raises(NotImplementedError):reward.step(step(1,(10000,9000),(event,)))
    assert reward.frame==0
    event['independent_input_segment']=False
    with pytest.raises(RuntimeError,match='duplicate'):reward.step(step(1,(10000,9000),(event,event)))
    with pytest.raises(RuntimeError,match='all HP'):reward.step(step(1,(10000,9000)))


def test_passive_cost_increment_no_refund_timeout():
    reward=CombatReward(CombatRewardConfig(combo_beta=0,counter_coefficient=0))
    reward.reset(step(0))
    total=0
    for frame in range(1,242):
        values,_=reward.step(step(frame,qualified_passive=(True,False)))
        total+=values[0]
    assert total==pytest.approx(-.02*(121/120)**2)
    values,_=reward.step(step(242,qualified_passive=(False,False)))
    assert values==(0.,0.)
    values,_=reward.step(step(243,qualified_passive=(False,False)),truncated=True)
    assert values==(-1.,-1.)


def test_technical_cap_and_block_chip_no_ch():
    reward=CombatReward(CombatRewardConfig(passive_b=0,technical_cap=10))
    reward.reset(step(0))
    event=hit(1,10000,9000);event['counter_hit_first']=True
    values,_=reward.step(step(1,(10000,9000),(event,)))
    assert values[0]==pytest.approx(.101)
    block=hit(2,9000,8990);block.update(kind='BLOCK_HIGH_RESOLUTION',hit_or_block='BLOCK')
    values,_=reward.step(step(2,(10000,8990),(block,)))
    assert values==pytest.approx((.001,-.001))


def test_missing_passive_evidence_is_not_inactivity_guess():
    reward=CombatReward(CombatRewardConfig())
    reward.reset(step(0))
    with pytest.raises(NotImplementedError,match='passive'):reward.step(step(1))


def test_diagnostic_mapping_cannot_authorize_formal_combo_bonus():
    reward=CombatReward(CombatRewardConfig(passive_b=0))
    reward.reset(step(0))
    event=hit(1,10000,9000)
    event.update(independent_input_segment=True,input_lineage_evidence='DIAGNOSTIC_ACTION_MAPPING')
    with pytest.raises(NotImplementedError,match='diagnostic action mapping'):
        reward.step(step(1,(10000,9000),(event,)))


def test_reward_contract_rejects_old_shaping_and_timeout_label():
    from soku_rl.env.hisouten_env import EpisodeConfig
    from soku_rl.env.match import LEGACY_MATCH
    from soku_rl.env.observation.visibility import VisibilityConfig
    from soku_rl.env.wrappers.learning import LearningInterface, LearningConfig
    from soku_rl.rl import validate_payoff
    config=EpisodeConfig(7200,8,1,0,'animation',VisibilityConfig(8,.5,.02,.1,48,96,16,.25),
        LEGACY_MATCH,combat_reward=dict(combo_beta=0,passive_b=0))
    with pytest.raises(ValueError,match='HP potential'):
        LearningInterface(config,LearningConfig('combat',False,8,1.))
    interface=LearningInterface(config,LearningConfig('combat',False,8,0.))
    with pytest.raises(ValueError,match='payoff'):
        validate_payoff(interface,dict(timeout_payoff='zero_at_horizon'))
    validate_payoff(interface,dict(timeout_payoff='native_combat_at_horizon'))


def test_unknown_lineage_is_explicitly_unpaid_not_relabeled_and_passive_is_bounded():
    config=CombatRewardConfig(unknown_combo_policy='no_bonus',passive_grace=2,passive_cap=25.)
    reward=CombatReward(config);reward.reset(step(0))
    event=hit(1,10000,9000);event['independent_input_segment']=None
    values,_=reward.step(step(1,(10000,9000),(event,),qualified_passive=(False,False),passive_reset=(True,True)))
    assert values==pytest.approx((.1,-.1))
    assert reward.totals[0]['unknown_lineage_hits']==1
    assert event['independent_input_segment'] is None
    for frame in range(2,30):
        reward.step(step(frame,(10000,9000),qualified_passive=(True,False),passive_reset=(False,False)))
    assert reward.passive_spent[0]==25.
    previous=reward.passive[0]
    reward.step(step(30,(10000,9000),qualified_passive=(False,False),passive_reset=(False,False)))
    assert reward.passive[0]==previous  # forced frames pause, not reset/refund

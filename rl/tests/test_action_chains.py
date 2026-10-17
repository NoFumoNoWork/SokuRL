"""Offline contracts: roots, causal states, gaps and no invented move onset."""
import numpy as np
import pytest

from soku_rl.replay.action_chains import extract_chains,decision_examples,input_program,weather_eligible
from sokurl_platform.command_input import CommandPlayer,Keys
from test_multiscale_bc import dense


def action(frame,root,action_id,order=1,accepted=True,scope=1):
    return dict(kind='ACTION_TRANSITION',frame=frame,segment=0,order=order,attacker_seat=0,
        attack_instance=root,action_before=0,action_after=action_id,accepted=accepted,
        lineage='DIRECT',native_input_scope=scope)


def hit(frame,root,count=0,order=1,lineage='CHILD'):
    return dict(kind='HIT_RESOLUTION',frame=frame,segment=0,order=order,attacker_seat=0,
        defender_seat=1,attack_instance=root,lineage=lineage,hit_or_block='HIT',
        combo_count_at_damage=count,combo_count_after=count+1,hp_before=10000,hp_after=9600)


def fixture():
    a=dense(12);a['raw_words'][:]=0
    records=[dict(frame=i+1,native_events=[]) for i in range(12)]
    return a,records


def test_actions_delayed_projectiles_and_multihit_share_root():
    a,r=fixture()
    r[1]['native_events']=[action(2,10,400)]
    r[2]['native_events']=[action(3,11,410)]
    r[4]['native_events']=[hit(5,10),hit(5,10,count=1,order=2)]
    r[6]['native_events']=[hit(7,11,count=2)]
    c=extract_chains(a,r,0)
    assert c['decisions'][0]['outcomes']['damage']==800
    assert c['combos'][0]['action_ids']==[400,410]
    assert c['combos'][0]['hits']==3 and c['combos'][0]['independent_actions']==2
    assert c['decisions'][0]['command_onset'] is None
    samples=decision_examples(a,c)
    mask=samples['kind']==1
    assert samples['observation_frames'][mask].tolist()==[1,2]
    assert 'outcomes' not in samples and 'next_action' not in samples
    np.testing.assert_array_equal(samples['observations'][mask],a['observations'][[1,2]])


def test_combo_reset_and_unknown_root_do_not_invent_segments():
    a,r=fixture();r[1]['native_events']=[action(2,10,400)]
    r[2]['native_events']=[hit(3,10)]
    r[3]['native_events']=[hit(4,999,count=1,lineage='UNKNOWN')]
    r[4]['native_events']=[hit(5,10,count=0)]
    c=extract_chains(a,r,0)
    assert len(c['combos'])==2
    assert c['combos'][0]['unknown_root_hits']==1
    assert c['contacts'][1]['decision_id'] is None


def test_automatic_transitions_and_no_contact_operations_retained():
    a,r=fixture();r[1]['native_events']=[action(2,1,400)]
    r[2]['native_events']=[action(3,2,401,scope=None)]
    c=extract_chains(a,r,0)
    assert len(c['decisions'])==1 and len(c['transitions'])==2
    assert c['decisions'][0]['outcomes']['hits']==0
    assert c['transitions'][1]['classification']=='NOT_PROVEN_INPUT_DECISION'


def test_same_frame_decisions_masked_not_falsely_ordered_states():
    a,r=fixture();r[2]['native_events']=[action(3,1,300),action(3,2,301,order=2)]
    c=extract_chains(a,r,0);s=decision_examples(a,c)
    assert not s['supervised'][s['kind']==1].any()


def test_gaps_do_not_join_movement_or_future_targets():
    a,r=fixture();a['raw_words_present'][4]=False
    r[1]['native_events']=[action(2,1,400)];r[6]['native_events']=[action(7,2,410)]
    c=extract_chains(a,r,0)
    assert c['decisions'][0]['next_decision'] is None
    assert all(not(m['start_row']<=4<=m['end_row']) for m in c['movements'])
    assert all(t.get('next') is None or not(t['observation_row']<4<
        t['observation_row']+t['next']['delta_frames']) for t in c['timeline'])


def test_movement_runs_are_event_weighted_not_hold_frame_weighted():
    a,r=fixture();a['raw_words'][:,0]=4
    c=extract_chains(a,r,0);s=decision_examples(a,c)
    assert len(c['movements'])==1 and len(s['kind'])==1
    assert tuple(s['movement'][0])==(-1,0,0)
    assert c['movements'][0]['accepted_movement']=='UNKNOWN'


def test_auto_phase_does_not_create_extra_hit_source():
    a,r=fixture();r[1]['native_events']=[action(2,10,400)]
    r[2]['native_events']=[action(3,11,401,scope=None)]
    r[4]['native_events']=[hit(5,10)]
    c=extract_chains(a,r,0)
    assert c['combos'][0]['action_ids']==[400]
    assert c['contacts'][0]['decision_id']==0


def test_alignment_rejected():
    a,r=fixture();r[1]['frame']=999
    with pytest.raises(ValueError):extract_chains(a,r,0)
    a,r=fixture();a['seats'][1]=1
    with pytest.raises(ValueError):extract_chains(a,r,0)
    a,r=fixture();r[1]['native_events']=[action(2,1,400,order=2),action(2,2,410,order=1)]
    with pytest.raises(ValueError):extract_chains(a,r,0)


def test_compiled_input_program_is_lossless_and_not_a_macro_guess():
    words=[0,4,4,4|16,0,8|128,8|128]
    program=input_program(words)
    from sokurl_platform.replay_inputs import decode_keys
    player=CommandPlayer([Keys(**stage) for stage in program])
    assert [player.next(i,1) for i in range(len(words))]==[tuple(decode_keys(w).values()) for w in words]


def test_weather_gate_keeps_cards_and_skill_conditions():
    from sokurl_platform.bridge_shared import WEATHER_CLEAR,WEATHER_TYPHOON
    assert weather_eligible(WEATHER_CLEAR)
    assert not weather_eligible(WEATHER_TYPHOON)
    assert not weather_eligible(WEATHER_CLEAR,cards_used=True)
    assert not weather_eligible(WEATHER_CLEAR,skills_default=False)


def test_future_hit_outcomes_do_not_enter_decision_inputs():
    a,r=fixture();r[1]['native_events']=[action(2,1,400)]
    before=decision_examples(a,extract_chains(a,r,0))
    r[8]['native_events']=[hit(9,1)]
    after=decision_examples(a,extract_chains(a,r,0))
    for key in before:np.testing.assert_array_equal(before[key],after[key])


def test_object_creation_requires_proven_lineage():
    a,r=fixture();r[1]['native_events']=[action(2,1,400)]
    r[2]['native_events']=[dict(kind='OBJECT_CREATED',frame=3,segment=0,order=1,
        attacker_seat=0,attack_instance=1,source_instance=99,lineage='UNKNOWN',action=800)]
    c=extract_chains(a,r,0)
    assert c['objects'][0]['decision_id'] is None
    assert c['decisions'][0]['outcomes']['objects']==0


def test_flight_root_retained_for_behavior_not_assumed_damage_move():
    a,r=fixture();r[1]['native_events']=[action(2,1,400)]
    r[2]['native_events']=[action(3,2,214)]
    r[4]['native_events']=[hit(5,1),hit(5,2,count=1,order=2)]
    combo=extract_chains(a,r,0)['combos'][0]
    assert combo['behavior_chain_eligible']
    assert not combo['damage_sources_in_p0_attack_range']
    assert combo['non_attack_root_actions']==[214]
    assert [s['action_id'] for s in combo['behavior_steps'] if s['kind']=='ACCEPTED_OPERATION']==[400,214]


def test_jump_between_attacks_retained_without_own_hit():
    a,r=fixture();r[1]['native_events']=[action(2,1,400)]
    r[2]['native_events']=[hit(3,1)]
    r[3]['native_events']=[action(4,2,208)]
    r[5]['native_events']=[action(6,3,410)]
    r[7]['native_events']=[hit(8,3,count=1)]
    combo=extract_chains(a,r,0)['combos'][0]
    assert combo['action_ids']==[400,410]
    steps=[s for s in combo['behavior_steps'] if s['kind']=='ACCEPTED_OPERATION']
    assert [s['action_id'] for s in steps]==[400,208,410]
    assert not steps[1]['contributed_contact']
    assert combo['behavior_chain_eligible']


def test_gap_blocks_behavior_linking_despite_known_hit_roots():
    a,r=fixture();r[1]['native_events']=[action(2,1,400)]
    r[2]['native_events']=[hit(3,1)]
    r[5]['native_events']=[action(6,2,410)]
    r[7]['native_events']=[hit(8,2,count=1)]
    a['compatible'][4]=False
    assert not extract_chains(a,r,0)['combos'][0]['behavior_chain_eligible']

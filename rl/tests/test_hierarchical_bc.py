"""Causality, masked supervision, typed architecture and fail-closed control."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from soku_rl.rl.animation_features import AnimationTokenFeatures
from soku_rl.rl.hierarchical_bc import HierarchicalActor,hierarchical_loss
from soku_rl.replay.hierarchical import PublicEventHistory,onset_label,build_hierarchical_case
from soku_rl.policy.hierarchical_control import command_catalog,Opportunity,EventCommandController,candidate_opportunity
from soku_rl.policy.hierarchical_bc import HierarchicalEpisode
from soku_rl.env.encoding import encode_action
from test_multiscale_bc import dense


def batch(actions,b=2):
    return dict(state=torch.zeros(b,185),event_states=torch.zeros(b,16,185),
        event_ids=torch.zeros(b,16,3,dtype=torch.long),event_numbers=torch.zeros(b,16,3),
        event_mask=torch.zeros(b,16,dtype=torch.bool),inputs=torch.zeros(b,60,8),
        candidates=torch.ones(b,actions,dtype=torch.bool),active_action=torch.full((b,),actions,dtype=torch.long),
        elapsed=torch.zeros(b))


@pytest.fixture
def model():
    torch.set_num_threads(2);torch.manual_seed(7)
    spatial=AnimationTokenFeatures(spaces.Box(-np.inf,np.inf,(1544,),np.float32),timing_vocab_size=1024)
    return HierarchicalActor(spatial,len(command_catalog())).eval()


def test_architecture_and_no_nan_with_empty_history(model):
    x=batch(model.actions);y=model(x)
    assert len(model.events.layers)==3
    assert model.events.layers[0].self_attn.num_heads==4
    assert y['action'].shape==(2,model.actions) and y['sequence'].shape==(2,3,model.actions)
    assert all(torch.isfinite(v).all() for v in y.values())
    with torch.inference_mode():
        for mask in (torch.zeros(2,16,dtype=torch.bool),torch.tensor([[0]*13+[1]*3,[0]*16],dtype=torch.bool)):
            x['event_mask']=mask
            assert all(torch.isfinite(v).all() for v in model(x).values())


def test_padded_event_states_cannot_affect_current_policy(model):
    x=batch(model.actions)
    with torch.inference_mode():
        a=model(x)['action'];x['event_numbers'][:]=123
        b=model(x)['action']
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_autoregression_is_shifted_and_does_not_leak_into_current(model):
    x=batch(model.actions);teacher=torch.tensor([[1,2,3],[2,3,4]])
    a=model(x,selected=torch.tensor([0,0]),teacher=teacher)
    changed=teacher.clone();changed[:,0]=7
    b=model(x,selected=torch.tensor([0,0]),teacher=changed)
    torch.testing.assert_close(a['action'],b['action'],rtol=0,atol=0)
    torch.testing.assert_close(a['sequence'][:,0],b['sequence'][:,0],rtol=0,atol=0)
    assert not torch.allclose(a['sequence'][:,1],b['sequence'][:,1])
    changed=teacher.clone();changed[:,2]=8
    c=model(x,selected=torch.tensor([0,0]),teacher=changed)
    torch.testing.assert_close(a['sequence'],c['sequence'],rtol=0,atol=0)


def test_masked_loss_weights_and_gradient(model):
    y=model(batch(model.actions),selected=torch.tensor([0,1]))
    targets=dict(action=torch.tensor([0,1]),action_mask=torch.tensor([True,True]),
        termination=torch.tensor([0.,1.]),termination_mask=torch.tensor([True,True]),
        sequence=torch.tensor([[1,2,-1],[2,3,4]]),sequence_mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool))
    losses=hierarchical_loss(y,targets)
    torch.testing.assert_close(losses['total'],losses['action']+.5*losses['termination']+.5*losses['sequence'])
    losses['total'].backward()
    assert model.events.layers[0].linear1.weight.grad.abs().sum()>0
    assert model.input_cnn[0].weight.grad is not None


def test_illegal_candidates_rejected_and_masked(model):
    x=batch(model.actions);x['candidates'][:]=False
    with pytest.raises(ValueError):model(x)
    x['candidates'][:,3]=True;y=model(x)
    assert (y['action'].argmax(-1)==3).all()


def test_public_history_never_reads_future_or_opponent_input():
    h=PublicEventHistory();state=np.zeros(185,np.float32);state[0]=state[10]=1
    h.observe(state,0)
    for i in range(1,61):h.observe(state,i,(0,)*8)
    a=h.tensors();state[5]=400;h.observe(state,61,(1,0,0,0,0,0,0,0))
    assert not a['event_mask'].any() and a['history_known']
    assert a['inputs'].sum()==0
    with pytest.raises(ValueError):h.observe(state,63)


def test_onset_is_before_native_acceptance_and_ambiguous_masked():
    a=dense(95);a['raw_words'][:]=0;a['observations'][:,1298]=1
    a['raw_words'][65,0]=16
    d=dict(action_id=300,observation_row=68)
    x=onset_label(a,d,{300:'5A'})
    assert x['onset']==65 and x['native_acceptance_row']==68
    a['raw_words'][67,0]=16
    assert onset_label(a,d,{300:'5A'})['onset'] is None


def test_dataset_input_excludes_current_and_future_keys():
    a=dense(95);a['raw_words'][:]=0;a['observations'][:,1298]=1;a['raw_words'][65,0]=16
    chains=dict(decisions=[dict(id=0,action_id=300,observation_row=68,same_frame_acceptances=1)],movements=[])
    data,audit=build_hierarchical_case(a,chains,{300:'5A'})
    i=next(i for i,s in enumerate(audit['samples']) if s['row']==65)
    assert data['target_action_mask'][i] and data['inputs'][i].sum()==0
    assert audit['samples'][i]['selection']['acceptance']==68


def opportunity(k,*,responsive=True,forced=False,source='PROVIDED_ENGINE_LEGAL_SET'):
    return Opportunity(responsive,(True,)*k,'PUBLIC_EVENT',source,forced)


def test_controller_can_interrupt_and_never_ignores_legal_unknown():
    catalog=command_catalog();c=EventCommandController(catalog,3);move=next(i for i,x in enumerate(catalog) if x['name']=='6move')
    op=opportunity(len(catalog));keys,replan=c.tick(0,1,0,[],0,op,decider=lambda _:move)
    assert keys[0]==1 and replan
    keys,replan=c.tick(1,1,0,[],0,op,stop=False,decider=lambda _:move)
    assert not replan and keys[0]==1  # Feedback opportunity alone is not forced switch.
    keys,replan=c.tick(2,1,0,[],0,op,stop=True,decider=lambda _:move)
    assert replan
    keys,replan=c.tick(3,1,0,[],0,opportunity(len(catalog),forced=True))
    assert keys==(0,)*8 and not replan
    with pytest.raises(RuntimeError):c.choose(move,opportunity(len(catalog),source='UNKNOWN'))


def test_episode_connects_selected_command_to_existing90_interface():
    catalog=command_catalog();move=next(i for i,x in enumerate(catalog) if x['name']=='6move')
    class Actor:
        def eval(self):return self
        def __call__(self,batch):
            logits=torch.full((1,len(catalog)),-100.);logits[:,move]=100.
            return dict(action=logits,termination=torch.tensor([-100.]))
    interface=SimpleNamespace(episode=SimpleNamespace(decision_frames=1,latency_frames=0),
        action_space=SimpleNamespace(n=90),inverse={encode_action((1,0,0,0,0,0,0,0)):70})
    controller=HierarchicalEpisode(Actor(),catalog,interface)
    observation=np.zeros(1544,np.float32)
    action,feedback=controller.act(observation,frame=0,facing=1,segment=0,events=[],seat=0,
        opportunity=opportunity(len(catalog)))
    assert action==70 and feedback['raw_logical_input']==(1,0,0,0,0,0,0,0)
    with pytest.raises(RuntimeError):controller.act(observation,frame=1,facing=1,segment=0,events=[],seat=0,
        opportunity=opportunity(len(catalog),source='CANDIDATES_ONLY'))


def test_unknown_operation_blocks_future_sequence_not_compressed():
    a=dense(110);a['raw_words'][:]=0;a['observations'][:,1298]=1
    a['raw_words'][65,0]=16;a['raw_words'][85,0]=32
    decisions=[dict(id=0,action_id=300,observation_row=65,same_frame_acceptances=1),
               dict(id=1,action_id=999,observation_row=75,same_frame_acceptances=1),
               dict(id=2,action_id=400,observation_row=85,same_frame_acceptances=1)]
    data,audit=build_hierarchical_case(a,dict(decisions=decisions,movements=[]),{300:'5A',400:'5B'})
    i=next(i for i,s in enumerate(audit['samples']) if s['row']==65)
    assert not data['target_sequence_mask'][i].any()
    assert any(x['status']=='UNKNOWN_COMMAND' for x in audit['unknown'])


def test_stop_duration_and_watchdog_are_not_repeated_decisions():
    catalog=command_catalog();index=next(i for i,x in enumerate(catalog) if x['name']=='6move')
    op=opportunity(len(catalog));c=EventCommandController(catalog,max_wait=3)
    flags=[]
    for frame in range(7):
        _,replan=c.tick(frame,1,0,[],0,op,decider=lambda _:index);flags.append(replan)
    assert flags==[True,False,False,True,False,False,True]


def test_no_input_gap_is_invented_as_neutral_history():
    a=dense(110);a['raw_words'][:]=0;a['observations'][:,1298]=1
    a['raw_words'][90,0]=16;a['raw_words_present'][80]=False
    data,audit=build_hierarchical_case(a,dict(decisions=[dict(id=0,action_id=300,
        observation_row=90,same_frame_acceptances=1)],movements=[]),{300:'5A'})
    assert not any(s['selection'] is not None for s in audit['samples'])


def test_passive_damage_is_context_not_action_label():
    a=dense(100);a['raw_words'][:]=0;a['observations'][:,1298]=1
    a['observations'][70:,1300]=50
    data,audit=build_hierarchical_case(a,dict(decisions=[],movements=[]),{})
    assert not data['target_action_mask'].any()
    assert not data['target_termination_mask'].any()


def test_history_cnn_order_and_features_are_used(model):
    x=batch(model.actions);x['inputs'][0,:,0]=torch.arange(60)/60
    with torch.inference_mode():
        a=model(x)['action'];x['inputs']=x['inputs'].flip(1);b=model(x)['action']
    assert not torch.allclose(a,b)


def test_completed_action_duration_only_appears_after_observed_end():
    history=PublicEventHistory();state=np.zeros(185,np.float32);state[0]=1;state[5]=400
    history.observe(state,0)
    for frame in range(1,6):history.observe(state,frame,(0,)*8)
    assert not any(e[0]==4 for e in history.events)
    state[5]=9;history.observe(state,6,(0,)*8)
    completed=[e for e in history.events if e[0]==4]
    assert completed[0][1]==400 and completed[0][4]==6


def test_native_feedback_mode_is_opt_in_and_never_marks_candidates_legal():
    catalog=command_catalog();index=next(i for i,e in enumerate(catalog) if e['name']=='5B')
    state=dict(action=0,hitstop=0,untech=0,frame_flags=0)
    op=candidate_opportunity(catalog,state,['5B','5move'])
    assert op.legality_source=='MASKED_CANDIDATES_NATIVE_AUTHORITY'
    assert sum(op.legal_actions)==2
    with pytest.raises(RuntimeError):EventCommandController(catalog).choose(index,op)
    c=EventCommandController(catalog,legality_mode='masked_native_feedback')
    c.tick(0,1,0,[],0,op,decider=lambda _:index)
    assert c.attempts[-1]['submitted'] and c.attempts[-1]['acceptance']=='NOT_OBSERVED'
    c.finish_attempt('TEST_END')
    assert not c.accepted and c.attempts[-1]['acceptance']=='NOT_OBSERVED'


def test_only_actual_native_feedback_can_confirm_attempt():
    catalog=command_catalog();index=next(i for i,e in enumerate(catalog) if e['name']=='5B')
    state=dict(action=0,hitstop=0,untech=0,frame_flags=0)
    op=candidate_opportunity(catalog,state,['5B'])
    c=EventCommandController(catalog,legality_mode='masked_native_feedback')
    c.tick(0,1,0,[],0,op,decider=lambda _:index)
    event=dict(kind='ACTION_TRANSITION',attacker_seat=0,accepted=True,lineage='DIRECT',
        native_input_scope=1,attack_instance=99,native_inputs_before=(0,0,0,1,0,0,0,0))
    c.tick(1,1,0,[event],0,op)
    assert c.accepted and c.attempts[-1]['acceptance']=='NATIVE_ACCEPTED'


def test_feedback_candidate_policy_blocks_forced_and_unknown_states():
    catalog=command_catalog()
    for state in (dict(action=50,hitstop=0,untech=0,frame_flags=0),
                  dict(action=0,hitstop=2,untech=0,frame_flags=0),
                  dict(action=999,hitstop=0,untech=0,frame_flags=0)):
        op=candidate_opportunity(catalog,state,['5B'])
        assert not op.can_respond and not any(op.legal_actions)

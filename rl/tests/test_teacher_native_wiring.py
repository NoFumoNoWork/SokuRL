"""Opt-in paired capture and event joins; synthetic fixtures, no game launch."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from game_runtime import observation as readers
from soku_rl.env.observation.shared_native import SCHEMA,assemble_shared
from soku_rl.policy.god.native_feedback import NativeTeacherFeedback
from test_teacher_continuous import teacher,packet,MASK,install_decision
from test_teacher_shared_native import native_example


def test_paired_reader_requires_same_paused_frame_and_redacts_private_data(monkeypatch):
    monkeypatch.setenv('SOKURL_COMBAT_EVENTS','0')
    import game_runtime.images as images
    import game_runtime.privileged as privileged
    closed=[]
    monkeypatch.setattr(images,'ImageClient',lambda pid:SimpleNamespace(
        read_state=lambda *args:None,close=lambda:closed.append('image')))
    monkeypatch.setattr(privileged,'ProcessMemory',lambda pid:None)
    native=native_example()
    monkeypatch.setattr(privileged,'SharedCurrentReader',lambda m:SimpleNamespace(
        observe=lambda *args:(native,native),close=lambda:closed.append('native')))
    monkeypatch.setattr(readers,'observe_animation',lambda *args:('a','b'))
    raw=SimpleNamespace(sceneId=5,battleMode=3,frameId=0,segmentId=1,
        p1=SimpleNamespace(hp=10000,characterId=1),p2=SimpleNamespace(hp=10000,characterId=0),
        stageId=12,activeWeather=21,stateHash=1,p1ObjectCount=0,p2ObjectCount=0)
    clock=SimpleNamespace(run_state_name='PAUSED',game_frame=0,
        latest=SimpleNamespace(segmentId=1),dropped_frames=0)
    bridge=SimpleNamespace(snapshot=lambda:clock)
    reader=readers.ObservationReader(1,'animation',None,shared_teacher_context=True)
    reader.combat_events=SimpleNamespace(observe=lambda raw:())
    try:
        result=reader.read(raw,bridge)
        view=result.diagnostics['shared_native_context'][0]
        assert view.players[1]['keys']==(0,)*10
        assert view.players[0]['deck']==(0,)*20
        clock.game_frame=1
        with pytest.raises(RuntimeError,match='exact paused'):
            reader.read(raw,bridge)
    finally:
        reader.close()
    assert closed==['native','image']


def test_opt_in_reader_never_reads_private_deck_and_default_is_unchanged():
    from game_runtime.privileged import SharedCurrentReader,PrivilegedReader
    from sokurl_platform.native_reader import PrivilegedReader as Original
    reader=SharedCurrentReader.__new__(SharedCurrentReader)
    reader.value=lambda *args:(_ for _ in ()).throw(AssertionError('must not read private deck'))
    assert reader.deck(0)==reader.deck(1)==(0,)*20
    assert PrivilegedReader.deck is Original.deck


@pytest.mark.parametrize('character,learner,script',[
    ('marisa',1,'01_marisa_main.ai'),('reimu',0,'00_reimu_main.ai')])
def test_bounded_native_smoke_profiles(character,learner,script):
    from smoke_teacher_native import smoke_profile
    profile=smoke_profile(character,1024,64)
    assert profile['learner']==learner and profile['opponent']==1-learner
    assert profile['script']==script and profile['windows']==(1024,64)


@pytest.mark.parametrize('character,frames,reset_frames',[
    ('unknown',128,64),('reimu',0,64),('marisa',1025,64),
    ('reimu',128,0),('marisa',128,129),('marisa',True,64)])
def test_native_smoke_rejects_unbounded_or_unknown_profile(character,frames,reset_frames):
    from smoke_teacher_native import smoke_profile
    with pytest.raises(ValueError):
        smoke_profile(character,frames,reset_frames)


def test_complete_episode_requires_explicit_original_horizon():
    from smoke_teacher_native import smoke_profile
    assert smoke_profile('reimu',7200,64,complete_episode=True)['windows']==(7200,64)
    with pytest.raises(ValueError):
        smoke_profile('marisa',7200,64)
    with pytest.raises(ValueError):
        smoke_profile('marisa',1024,64,complete_episode=True)


def test_terminal_slots_allow_subset_progress_without_auto_reset():
    from smoke_teacher_native import completed_slots
    from soku_rl.env.encoding import AGENTS
    flags=lambda value:dict.fromkeys(AGENTS,value)
    assert completed_slots({0:flags(True),1:flags(False)},{0:flags(False),1:flags(False)})=={0}
    assert completed_slots({1:flags(False)},{1:flags(True)})=={1}
    with pytest.raises(ValueError):
        completed_slots({0:flags(True)},{0:flags(True)})
    with pytest.raises(ValueError):
        completed_slots({0:{AGENTS[0]:True,AGENTS[1]:False}},{0:flags(False)})


@pytest.mark.parametrize('script',['01_marisa_main.ai','00_reimu_main.ai'])
def test_terminal_receipt_closes_without_running_next_input_and_requires_new_actor(script):
    from soku_rl.policy.god.continuous import ExecutionReceipt
    actor=teacher(script)
    install_decision(actor,b'key_reset();key_on(ACT_B);yield();error("MAIN_RESUMED_AFTER_TERMINAL")')
    first=actor.query(packet(0,script),schema=SCHEMA,trusted_boundary=True,legal_actions=MASK)
    receipt=ExecutionReceipt(0,first['training_label'],1,0,True,True,
        accepted_plan_id=actor.serial,accepted_buttons=('ACT_B',))
    result=actor.query(packet(1,script),schema=SCHEMA,receipt=receipt,terminal=True)
    assert result['training_label'] is None and result['reason']=='EPISODE_TERMINAL'
    assert result['reconciliation']['stage']=='ACCEPTED'
    assert actor.counts['accepted']==1 and actor.counts['observations']==2
    assert actor.ended and actor.runner is actor.pending is None
    assert actor.counts['terminal_closures']==1
    with pytest.raises(ValueError,match='new teacher at Reset'):
        actor.query(packet(2,script),schema=SCHEMA,legal_actions=MASK,trusted_boundary=True)
    fresh=teacher(script)
    assert not fresh.ended and fresh.counts['queries']==0 and fresh.serial==0


def test_terminal_flag_is_explicit_and_invalid_flag_does_not_advance_teacher():
    actor=teacher()
    with pytest.raises(ValueError,match='terminal flag'):
        actor.query(packet(0),schema=SCHEMA,terminal=1)
    assert actor.frame==-1 and not actor.ended


def events(frame,keys,*,scope=True):
    common=dict(frame=frame,segment=7,attacker_seat=0,native_inputs_before=keys,
        buffered_inputs_before=(0,)*8)
    result=[dict(common,kind='NATIVE_INPUT_SAMPLE',order=3,native_input_scope=0)]
    if scope:result.insert(0,dict(common,kind='NATIVE_INPUT_SCOPE',order=1,native_input_scope=12))
    return result


def test_frame_end_sample_alone_is_not_consumption():
    actor=teacher()
    join=NativeTeacherFeedback(actor.interface,0,7,initial_facing=1)
    current=native_example(1)
    receipt,boundary,evidence=join.observe(1,events(1,(0,)*8,scope=False),actor.initial_action,actor,current)
    assert not receipt.consumed and receipt.poll_index is None and not boundary
    assert evidence['native_frame']==1 and receipt.frame==0


def test_join_keeps_scope_identity_and_independent_receipt_sequence():
    actor=teacher()
    join=NativeTeacherFeedback(actor.interface,0,7,initial_facing=1)
    receipt,boundary,evidence=join.observe(1,events(1,(0,)*8),actor.initial_action,actor,native_example(1))
    assert receipt.consumed and receipt.poll_index==1 and boundary
    assert evidence['native_scope_ids']==[12]  # not replaced by frame or ordinal
    with pytest.raises(ValueError,match='frame/segment'):
        join.observe(2,events(3,(0,)*8),actor.initial_action,actor,native_example(2))


def test_default_and_shared_layout_contracts_are_explicit():
    from soku_rl.env.hisouten_env import EpisodeConfig
    from soku_rl.env.wrappers.learning import LearningConfig,LearningInterface
    from soku_rl.env.match import LEGACY_MATCH
    from soku_rl.env.observation.visibility import VisibilityConfig
    visibility=VisibilityConfig(8,.5,.02,.1,48.,96.,16.,.25)
    base=EpisodeConfig(7200,8,1,0,'animation',visibility,LEGACY_MATCH,
        animation_world_coordinates=True)
    config=LearningConfig('combat',False,8,0.)
    assert LearningInterface(base,config).observation_space.shape==(1832,)
    native=replace(base,shared_teacher_context=True)
    assert LearningInterface(native,config).observation_space.shape==(592658,)
    assert 'shared_teacher_context' not in base.backend_observation()
    assert native.backend_observation()['shared_teacher_context']
    with pytest.raises(ValueError):
        replace(native,decision_frames=2)


@pytest.mark.parametrize('ambiguous',[False,True])
def test_native_acceptance_reuses_platform_direct_scope_rule(ambiguous):
    from soku_rl.env.encoding import encode_action
    actor=teacher();actor.serial=1
    inputs=(0,0,0,1,0,0,0,0)
    action=actor.interface.action(encode_action(inputs))
    actor.pending={'candidate':action}
    stream=events(1,inputs)
    stream.insert(1,dict(frame=1,segment=7,order=2,attacker_seat=0,
        kind='ACTION_TRANSITION',native_input_scope=12,accepted=True,lineage='DIRECT',
        attack_instance=42,native_inputs_before=inputs,released_inputs_before=(0,)*6,
        buffered_inputs_before=(0,0,int(ambiguous),0,0,0,0,0)))
    join=NativeTeacherFeedback(actor.interface,0,7,initial_facing=1)
    receipt,_,evidence=join.observe(1,stream,action,actor,native_example(1))
    assert receipt.accepted is (not ambiguous)
    assert evidence['accepted_roots']==([] if ambiguous else [42])


@pytest.mark.parametrize('script',['01_marisa_main.ai','00_reimu_main.ai'])
@pytest.mark.parametrize('helper',['lvr236','lvr623','lvr236h'])
def test_original_command_to_receipt_to_teacher_is_integrated_synthetic_only(script,helper):
    """Original input program,synthetic state/events;NOT original-game acceptance."""
    from soku_rl.env.encoding import decode_action
    actor=teacher(script)
    install_decision(actor,('key_reset();'+helper+'(ACT_B);wait(3);key_reset();wait(2)').encode())
    join=NativeTeacherFeedback(actor.interface,0,7,initial_facing=1)
    receipt=None
    accepted=0
    emitted=[]
    # Include the query after release:the last step's receipt is only reconciled
    # on the following decision,not when the input is first submitted.
    for frame in range(8):
        advice=actor.query(packet(frame,script),schema=SCHEMA,receipt=receipt,
            trusted_boundary=frame==0,legal_actions=MASK)
        action=advice['training_label']
        assert action is not None,advice
        keys=decode_action(actor.interface.command(action)).inputs
        emitted.append(keys)
        stream=events(frame+1,keys)
        if keys[3] and not accepted:
            stream.insert(1,dict(frame=frame+1,segment=7,order=2,attacker_seat=0,
                kind='ACTION_TRANSITION',native_input_scope=12,accepted=True,lineage='DIRECT',
                attack_instance=42,native_inputs_before=keys,released_inputs_before=(0,)*6,
                buffered_inputs_before=(0,)*8))
            accepted+=1
        current=native_example(frame+1)
        current.players[0]['char']=0 if script.startswith('00_') else 1
        receipt,_,_=join.observe(frame+1,stream,action,actor,current)
    assert accepted==actor.counts['accepted']==1
    assert actor.serial==1 and actor.counts['invalidations']==0
    assert emitted[0][:2]==((0,1) if helper.startswith('lvr236') else (1,0))
    assert actor.counts['input:pressed:B']==1
    assert actor.counts['input:released:B']==1

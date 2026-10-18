"""Opt-in paired capture and event joins; synthetic fixtures, no game launch."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from game_runtime import observation as readers
from soku_rl.env.observation.shared_native import SCHEMA,assemble_shared
from soku_rl.policy.god.native_feedback import NativeTeacherFeedback
from test_teacher_continuous import teacher,packet,MASK
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

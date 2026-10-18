"""Same sanitized tensor for learner/teacher; no new native reads in these tests."""
import copy
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from soku_rl.env.observation.memory_schema import FIGHTER_NAMES,OBJECT_NAMES,WORLD_NAMES,MAX_OBJECTS
from soku_rl.env.observation.privileged import PrivilegedObservation,encode_privileged
from soku_rl.env.observation.shared_native import (SCHEMA,PUBLIC_FEATURES,SHARED_FEATURES,
    SharedNativeContract,assemble_shared,decode_shared)
from soku_rl.policy.god.shared_native import SharedTeacherPerception,SharedNativeSourceActor
from soku_rl.rl.shared_native_features import SharedNativeFeatures,initialize_shared_policy
from soku_rl.rl.animation_features import AnimationTokenFeatures


def native_example(frame=0):
    """Complete numeric SYNTHETIC fixture, never claimed as a captured game."""
    players=[]
    for seat in (0,1):
        player=dict.fromkeys(FIGHTER_NAMES,0)
        player.update(char=1-seat,x=320+480*seat,y=0,dir=1-2*seat,hp=10000,
            xspeed=1.25,yspeed=-0.5,rei=4321,rmax=5000,frame=7,fflags=0x80000040,
            aflags=0x40008000,address=0x12345000+seat*0x1000,
            skills=(0,1,2,4)+(-1,)*12,cards=(-1,)*10,special=(0,)*28,
            keys=(13,)*10,deck=tuple(range(20)),
            hitarea=((0,0,20,100),),attackarea=(),hitarea_n=1,attackarea_n=0,obj_n=1)
        obj=dict.fromkeys(OBJECT_NAMES,0)
        obj.update(act=800,img=196,x=500,y=0,dir=1-2*seat,frame=40,hp=1,
            xspeed=3.5,fflags=0x80000001,aflags=0x40000001,address=0x23456000+seat*0x100,
            hitarea=(),attackarea=((480,0,520,20),),hitarea_n=0,attackarea_n=1)
        player['objects']=(obj,)
        players.append(player)
    world=dict.fromkeys(WORLD_NAMES,0)
    world.update(frame=frame,battle_time=frame,weather=21,weather2=21,stage_number=12,bgm_number=0)
    return PrivilegedObservation(world,tuple(players))


def public_example(native=None):
    native=native_example() if native is None else native
    row=np.zeros(221,np.float32)
    row[0]=row[10]=1;row[3]=1;row[13]=-1;row[4]=1
    row[180:184]=(1,1,1,1)
    row[185:189]=(320,0,800,0)
    for seat,player in enumerate(native.players):
        start=seat*10
        for column,name in ((3,'dir'),(4,'char'),(5,'act'),(6,'act_block'),(9,'frame')):
            row[start+column]=player[name]
        row[185+seat*2:187+seat*2]=(player['x'],player['y'])
    return np.r_[np.tile(row,8),np.zeros(64,np.float32)]


def test_common_tensor_exact_flags_skills_boxes_objects_and_redaction():
    original=native_example();before=copy.deepcopy(original)
    public=public_example()
    shared=assemble_shared(public,original,public_frame=0)
    decoded=decode_shared(shared,schema=SCHEMA)
    assert np.array_equal(shared[:PUBLIC_FEATURES],public)
    for seat in (0,1):
        p=decoded.players[seat]
        for name in FIGHTER_NAMES: assert p[name]==original.players[seat][name]
        assert p['skills']==original.players[seat]['skills']
        assert p['objects']==original.players[seat]['objects']
        assert p['hitarea']==original.players[seat]['hitarea']
        assert p['keys']==((13,)*10 if seat==0 else (0,)*10) and p['deck']==(0,)*20
    assert original==before  # no in-place sanitization or reader mutation
    assert SharedNativeContract().report()['additional_reaction_frames']==0


def test_private_keys_future_deck_and_unknown_annotations_cannot_leak():
    original=native_example();changed=copy.deepcopy(original)
    changed.players[1]['keys']=(1000,)*10
    changed.players[0]['deck']=tuple(reversed(range(20)))
    changed.players[1]['next_action']=400
    changed.world['future_rng']=123
    assert np.array_equal(assemble_shared(public_example(),original,public_frame=0),
                          assemble_shared(public_example(),changed,public_frame=0))
    # A caller cannot bypass sanitization and submit a raw privileged payload.
    injected=np.r_[public_example(),encode_privileged(original)]
    with pytest.raises(ValueError,match='redacted'): decode_shared(injected,schema=SCHEMA)


def test_wrong_frame_version_incomplete_object_lists_and_nonfinite_are_rejected():
    original=native_example()
    with pytest.raises(ValueError,match='frame mismatch'):
        assemble_shared(public_example(),original,public_frame=1)
    wrong=public_example();wrong[7*221+5]=400
    with pytest.raises(ValueError,match='fighter state mismatch'):
        assemble_shared(wrong,original,public_frame=0)
    shared=assemble_shared(public_example(),original,public_frame=0)
    with pytest.raises(ValueError,match='version'): decode_shared(shared,schema='old1832')
    original.players[1]['obj_n']=2
    with pytest.raises(ValueError,match='object observation'): assemble_shared(public_example(),original,public_frame=0)
    invalid=public_example();invalid[0]=np.nan
    with pytest.raises(ValueError): assemble_shared(invalid,native_example(),public_frame=0)
    overflow=native_example()
    overflow.players[0]['objects']=(overflow.players[0]['objects'][0],)*(MAX_OBJECTS+1)
    overflow.players[0]['obj_n']=MAX_OBJECTS+1
    with pytest.raises(ValueError,match='object observation'):
        assemble_shared(public_example(),overflow,public_frame=0)


def test_native_feature_path_drops_pointer_magnitude_not_game_state():
    contract=gym.spaces.Box(-np.inf,np.inf,(SHARED_FEATURES,),np.float32)
    feature=SharedNativeFeatures(contract,public_kwargs=dict(width=16,heads=4),features_dim=32)
    # Exercise a NONZERO context so pointer invariance is not merely the initial
    # zero-output migration guarantee.
    torch.nn.init.normal_(feature.native.output[0].weight,std=.01)
    first=native_example();second=copy.deepcopy(first)
    for player in second.players:
        player['address']+=0x100000
        for obj in player['objects']:obj['address']+=0x200000
    a=assemble_shared(public_example(),first,public_frame=0)
    b=assemble_shared(public_example(),second,public_frame=0)
    with torch.no_grad():
        x=feature(torch.as_tensor(a[None]))
        y=feature(torch.as_tensor(b[None]))
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert not np.array_equal(a,b)  # common packet retains exact opaque IDs


def test_teacher_has_exact_same_current_fields_and_no_private_key_api():
    teacher=SharedTeacherPerception(17)
    shared=assemble_shared(public_example(),native_example(),public_frame=0)
    current=teacher.observe(shared,schema=SCHEMA)
    assert current==decode_shared(shared,schema=SCHEMA)
    assert teacher.api.get_skill_lv(0,3)==4
    assert teacher.api.get_fflags(0,0x80000000)
    assert teacher.api.get_aflags(1,32768)
    assert teacher.api.get_obj_data(1,0)[4]==196
    assert teacher.api.get_obj_attackarea(1,0,0)==(480,0,520,20)
    assert teacher.api.get_key_stat2(0,4)==13
    assert teacher.api.get_key_map2(0)==(13,)*10
    for name,args in (('get_key_stat2',(1,4)),('get_key_map2',(1,)),('get_deck_list',(0,))):
        with pytest.raises(ValueError,match='UNAVAILABLE'): getattr(teacher.api,name)(*args)
    with pytest.raises(ValueError,match='consecutive'): teacher.observe(shared,schema=SCHEMA)
    fresh=SharedTeacherPerception(18)
    fresh.observe(shared,schema=SCHEMA)
    assert fresh.api is not teacher.api and fresh.frame==0


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_student_old_outputs_transfer_gradients_and_save_restore(device,tmp_path):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    public_kwargs=dict(object_slots=16,world_coordinates=True,width=16,heads=4,features_dim=32)
    old=PPO('MlpPolicy',Dummy(PUBLIC_FEATURES),device=device,
        n_steps=2,batch_size=2,policy_kwargs=dict(features_extractor_class=AnimationTokenFeatures,
            features_extractor_kwargs=public_kwargs,net_arch=dict(pi=[16],vf=[16])))
    new=PPO('MlpPolicy',Dummy(SHARED_FEATURES),device=device,n_steps=2,batch_size=2,
        policy_kwargs=dict(features_extractor_class=SharedNativeFeatures,
            features_extractor_kwargs=dict(public_kwargs=public_kwargs,features_dim=32),
            net_arch=dict(pi=[16],vf=[16])))
    original={k:v.detach().clone() for k,v in old.policy.state_dict().items()}
    metadata=initialize_shared_policy(new.policy,old.policy)
    shared=assemble_shared(public_example(),native_example(),public_frame=0)
    a=torch.as_tensor(public_example()[None],device=device)
    b=torch.as_tensor(shared[None],device=device)
    old.policy.set_training_mode(False);new.policy.set_training_mode(False)
    with torch.no_grad():
        before=old.policy.get_distribution(a).distribution.probs
        after=new.policy.get_distribution(b).distribution.probs
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        torch.testing.assert_close(old.policy.predict_values(a),new.policy.predict_values(b),rtol=0,atol=0)
    assert all(torch.equal(v,old.policy.state_dict()[k]) for k,v in original.items())
    assert metadata['optimizer']=='fresh'
    loss=-new.policy.get_distribution(b).distribution.log_prob(torch.tensor([0],device=device)).mean()
    loss.backward()
    assert new.policy.features_extractor.native.output[0].weight.grad.abs().sum()>0
    new.save(tmp_path/'shared.zip')
    loaded=PPO.load(tmp_path/'shared.zip',device=device)
    loaded.policy.set_training_mode(False)
    with torch.no_grad():
        torch.testing.assert_close(after,loaded.policy.get_distribution(b).distribution.probs,rtol=0,atol=0)


class Dummy(gym.Env):
    def __init__(self,width):
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(width,),np.float32)
        self.action_space=gym.spaces.Discrete(90)


@pytest.mark.parametrize('script',['01_marisa_main.ai','00_reimu_main.ai'])
def test_original_source_runs_with_common_fields_only_in_controlled_diagnostic(script):
    from soku_rl.policy.god.runtime import GodPolicy
    from soku_rl.policy.god.package import ScriptPackage
    root=Path(__file__).resolve().parents[2]
    api=root/'third_party/th123_ai/source/th123_ai/api.ai'
    if not api.is_file():pytest.skip('external2011source unavailable')
    package=ScriptPackage(root/'third_party/th123_ai/package/th123ai/script',api)
    policy=GodPolicy('shared-diagnostic',package,script,
        SimpleNamespace(observation_mode='privileged_state',decision_frames=1,latency_frames=0,history_frames=1))
    actor=SharedNativeSourceActor(policy,17)
    previous=None
    def snapshot(frame):
        data=native_example(frame)
        data.players[0]['char']=1 if script.startswith('01_') else 0
        data.players[1]['char']=1-data.players[0]['char']
        return data
    for frame in range(8):
        # Controlled test feedback, NOT an observed native execution/acceptance.
        packet=assemble_shared(public_example(snapshot(frame)),snapshot(frame),public_frame=frame)
        advice=actor.propose(packet,schema=SCHEMA,previous_executed=previous)
        assert advice['status']=='CONTROLLED_SOURCE_PROPOSAL' and advice['training_label'] is None
        previous=advice['command']
    packet=assemble_shared(public_example(snapshot(8)),snapshot(8),public_frame=8)
    conflict=actor.propose(packet,schema=SCHEMA,previous_executed=previous^1)
    assert conflict['status']=='BLOCKED_CONTROL_ALIGNMENT'
    assert actor.last_frame==7  # no stale coroutine was resumed

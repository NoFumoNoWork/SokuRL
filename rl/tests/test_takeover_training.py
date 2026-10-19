"""No native game: schedule, data ownership, optimizer isolation and recovery."""
import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from soku_rl.rl.masked_policy import CardMaskedActorCriticPolicy
from soku_rl.rl.takeover_training import (TakeoverSchedule,Demonstrations,ActorBC,
    save_bundle,restore_bundle,load_student)


class Toy(gym.Env):
    observation_space=gym.spaces.Box(-10,10,(4,),np.float32)
    action_space=gym.spaces.Discrete(90)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);self.t=0
        return np.zeros(4,np.float32),{}
    def step(self,action):
        self.t+=1
        return np.array([self.t%3,0,0,0],np.float32),float(action==2),self.t==5,False,{}


def student():
    torch.set_num_threads(2)
    return PPO(CardMaskedActorCriticPolicy,Toy(),n_steps=8,batch_size=4,n_epochs=2,
        policy_kwargs=dict(masked_actions=(89,),net_arch=[8,8]),device='cpu',seed=4)


def add(schedule,outcomes):
    for i,outcome in enumerate(outcomes):
        schedule.add(str(i),outcome,teacher_frames=0,policy_hash='frozen')


def test_new_formula_and_longest_streak_not_trailing():
    s=TakeoverSchedule();assert s.snapshot()['alpha']==1
    s.add('first','win',teacher_frames=0,policy_hash='p')
    assert s.snapshot()['alpha']==pytest.approx(.4)
    s=TakeoverSchedule();add(s,['win','win','loss','win','loss','win','loss','win','loss','loss'])
    assert s.snapshot()==dict(t=10,n=10,wins=5,longest_streak=2,alpha=.55)
    add_all=TakeoverSchedule();add(add_all,['win']*5)
    assert add_all.snapshot()['alpha']==0


def test_window_rolls_and_nonwins_break_streak():
    s=TakeoverSchedule();add(s,['win']*5+['loss']*10)
    assert s.snapshot()==dict(t=15,n=10,wins=0,longest_streak=0,alpha=1.)
    s=TakeoverSchedule();add(s,['win','draw','win','timeout','win'])
    assert s.snapshot()['longest_streak']==1
    with pytest.raises(ValueError):s.add('bad','win',teacher_frames=1,policy_hash='p')
    with pytest.raises(ValueError):s.add('0','win',teacher_frames=0,policy_hash='p')


def test_demo_masks_and_memory_bounds_do_not_alter_observation():
    data=Demonstrations(2,4096)
    x=np.array([0.,-0.,2.,3.],np.float32)
    assert not data.add(x,None,actor_source='student_fallback',valid=False,context={})
    assert not data.add(x,2,actor_source='student',valid=True,context={})
    for i in range(3):assert data.add(x,2,actor_source='teacher',valid=True,context={'frame':i})
    assert data.evicted==1 and [r['context']['frame'] for r in data.rows]==[1,2]
    a,_,_=data.sample(np.random.default_rng(0),2,'cpu')
    assert np.array_equal(a.numpy()[0].view('u4'),x.view('u4'))
    with pytest.raises(ValueError):data.add(x,90,actor_source='teacher',valid=True,context={})


def test_bc_updates_same_actor_not_critic_or_adam_or_ppo_buffer():
    model=student();model.learn(8)
    model.rollout_buffer.reset()
    critic={k:v.clone() for k,v in model.policy.state_dict().items() if 'value_net' in k}
    actor=model.policy.action_net.weight.detach().clone()
    adam=copy.deepcopy(model.policy.optimizer.state_dict())
    before_steps=model.num_timesteps;updates=model._n_updates
    bc=ActorBC(model,1.,.05)
    obs=torch.zeros(16,4);target=torch.full((16,),2,dtype=torch.long)
    model.policy.set_training_mode(False)
    before=-model.policy.get_distribution(obs).log_prob(target).mean().item()
    for _ in range(10):bc.step(obs,target)
    model.policy.set_training_mode(False)
    after=-model.policy.get_distribution(obs).log_prob(target).mean().item()
    assert after<before
    assert not torch.equal(actor,model.policy.action_net.weight)
    assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in critic.items())
    assert model.num_timesteps==before_steps and model._n_updates==updates
    assert model.rollout_buffer.pos==0 and not model.rollout_buffer.full
    now=model.policy.optimizer.state_dict()
    assert now['param_groups']==adam['param_groups']
    assert all(torch.equal(v,now['state'][key][name]) for key,entry in adam['state'].items() for name,v in entry.items())


def test_ppo_still_updates_actor_and_critic_after_bc():
    model=student();bc=ActorBC(model,1.,.01)
    bc.step(torch.zeros(4,4),torch.full((4,),2,dtype=torch.long))
    actor=model.policy.action_net.weight.detach().clone()
    critic=model.policy.value_net.weight.detach().clone()
    model.learn(16)
    assert model._n_updates==4
    assert not torch.equal(actor,model.policy.action_net.weight)
    assert not torch.equal(critic,model.policy.value_net.weight)


def test_checkpoint_keeps_history_buffer_cursor_rng_and_next_bc_update(tmp_path):
    model=student();model.learn(8)
    bc=ActorBC(model,1.,.01)
    data=Demonstrations(8,8192)
    data.add(np.ones(4,np.float32),2,actor_source='teacher',valid=True,context={'frame':0})
    s=TakeoverSchedule();add(s,['win','loss'])
    state=dict(phase='bc',cycle=0,bc_cycle_updates=2,evaluations=s.history,demos=data,
        rng=np.random.default_rng(5).bit_generator.state)
    identity={'contract':'same_actor_two_streams'}
    path=save_bundle(tmp_path/'checkpoint',model,bc,state,identity)
    obs=torch.zeros(4,4);labels=torch.full((4,),2,dtype=torch.long)
    bc.step(obs,labels)
    expected={k:v.clone() for k,v in model.policy.state_dict().items()}
    restored,other,saved=restore_bundle(path,Toy(),'cpu',identity,1.,.01)
    assert TakeoverSchedule(saved['evaluations']).snapshot()==s.snapshot()
    assert saved['bc_cycle_updates']==2 and saved['phase']=='bc'
    assert saved['demos'].next_id==1 and len(saved['demos'].rows)==1
    other.step(obs,labels)
    assert all(torch.equal(v,restored.policy.state_dict()[k]) for k,v in expected.items())
    assert other.updates==bc.updates
    with pytest.raises(ValueError):restore_bundle(path,Toy(),'cpu',{'wrong':1},1.,.01)


def test_explicit_load_selects_upstream_ppo_and_preserves_weights(tmp_path):
    model=student();model.learn(8)
    model.kickstarting_state={'old':'not an active loss'}
    model.save(tmp_path/'source.zip')
    loaded=load_student(tmp_path/'source.zip',Toy(),'cpu')
    assert type(loaded) is PPO and not hasattr(loaded,'kickstarting_state')
    assert loaded.num_timesteps==model.num_timesteps
    assert all(torch.equal(v,loaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())


def test_cpu_restart_is_explicit_and_only_replaces_requested_slots(monkeypatch):
    from native_cpu_worker import NativeCPUTrainingBatch,SokuGameBatch
    backend=object.__new__(NativeCPUTrainingBatch)
    backend.processes={0:object(),1:object()}
    closed=[]
    backend._close_slots=lambda slots:closed.append(slots)
    monkeypatch.setattr(SokuGameBatch,'reset_slots',lambda self,seeds:seeds)
    monkeypatch.setenv('SOKURL_CPU_RESET_PROCESS_RESTART','1')
    assert backend.reset_slots({1:123})=={1:123}
    assert closed==[{1}]
    monkeypatch.delenv('SOKURL_CPU_RESET_PROCESS_RESTART')
    backend.reset_slots({0:123})
    assert closed==[{1}]


def test_cpu_fresh_rejects_conflicting_restart_before_launch(monkeypatch):
    from native_cpu_worker import NativeCPUTrainingBatch
    backend=object.__new__(NativeCPUTrainingBatch)
    backend.fresh_reset=True;backend.recording_enabled=False;backend.fixed_weather='clear'
    monkeypatch.setenv('SOKURL_NATIVE_CPU_FRESH_RESET','1')
    monkeypatch.setenv('SOKURL_CPU_RESET_PROCESS_RESTART','1')
    with pytest.raises(ValueError,match='mutually exclusive'):backend._launch_slots({0:123})


@pytest.mark.parametrize('label,interval,expected_calls',[(3,0,0),(3,1,1),(None,0,1)])
def test_auxiliary_lazy_proposal_only_skips_trusted_teacher_frames(tmp_path,monkeypatch,label,interval,expected_calls):
    import train_takeover as tool
    from contextlib import contextmanager
    from types import SimpleNamespace
    from dataclasses import dataclass
    @dataclass
    class Receipt:
        consumed:bool=True
    class Teacher:
        def __init__(self,*a,**kw):pass
        def query(self,*a,**kw):return {'training_label':label}
        def report(self):return {}
    class Feedback:
        def __init__(self,*a,**kw):pass
        def observe(self,*a):return Receipt(),True,{}
    class View:
        reset_infos=[{'segment':0}]
        def reset(self):return np.zeros((1,4),np.float32)
        def step(self,actions):
            assert actions[0]==(label if label is not None else 2)
            return self.reset(),[0],[True],[dict(episode_id=0,native_combat_events=[],
                terminal_observation=np.zeros(4,np.float32),source_truncated=False,outcome='p1_win')]
    @contextmanager
    def game(*a):yield View(),SimpleNamespace(action=lambda a:a),None
    model=SimpleNamespace(policy=SimpleNamespace(masked_actions=()),num_timesteps=8,_n_updates=1,
        rollout_buffer=SimpleNamespace(pos=0,full=False))
    calls=[]
    model.predict=lambda *a,**kw:(calls.append(1) or np.array([2]),None)
    monkeypatch.setattr(tool,'game_view',game)
    monkeypatch.setattr(tool,'ContinuousSourceTeacher',Teacher)
    monkeypatch.setattr(tool,'NativeTeacherFeedback',Feedback)
    monkeypatch.setattr(tool,'ScriptPackage',lambda *a:None)
    monkeypatch.setattr(tool,'decode_shared',lambda *a,**kw:SimpleNamespace(players=[{'dir':1}]))
    monkeypatch.setattr(tool,'parameter_hash',lambda *a:'unchanged')
    buffer=Demonstrations(8,8192)
    result=tool.auxiliary_episode(model,{'episode':{'max_frames':1}},
        dict(package='',api_source='',script='',instruction_limit=1),tmp_path,{},1,True,buffer,1,interval)
    assert len(calls)==expected_calls and result['ppo_buffer_frames_added']==0
    assert len(buffer.rows)==(1 if label is not None else 0)

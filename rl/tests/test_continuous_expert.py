"""Continuous on-policy attachment contracts;synthetic packets unless noted."""
import gymnasium as gym
import numpy as np
import pytest
import torch

from soku_rl.env.observation.shared_native import SCHEMA,SHARED_FEATURES,decode_shared,assemble_shared
from soku_rl.rl.continuous_expert import ContinuousExpert,validated_native_source,validate_spec
from soku_rl.rl.kickstarting import PackedExpertRolloutBuffer
from soku_rl.rl.storage import PackedArray
from test_teacher_continuous import packet
from test_teacher_native_wiring import events
from test_teacher_shared_native import public_example


def spec():
    return dict(provider='continuous_shared_v1',package='third_party/th123_ai/package/th123ai/script',
        api_source='third_party/th123_ai/source/th123_ai/api.ai',script='01_marisa_main.ai',
        seed=17,instruction_limit=200000,
        validation='rl/runs/teacher-episode-20261019/marisa-v2/run/report.json')


def interface():
    from soku_rl.env.hisouten_env import EpisodeConfig
    from soku_rl.env.match import LEGACY_MATCH
    from soku_rl.env.observation.visibility import VisibilityConfig
    from soku_rl.env.wrappers.learning import LearningConfig,LearningInterface
    cfg=EpisodeConfig(7200,8,1,0,'animation',VisibilityConfig(8,.5,.02,.1,48.,96.,16.,.25),
        LEGACY_MATCH,animation_world_coordinates=True,shared_teacher_context=True)
    return LearningInterface(cfg,LearningConfig('combat',False,8,0.))


def make_expert(monkeypatch):
    import soku_rl.rl.continuous_expert as module
    # Unit fixtures deliberately bypass native gate;not native verification.
    monkeypatch.setattr(module,'validated_native_source',lambda spec:'SYNTHETIC_ONLY')
    return ContinuousExpert(interface(),spec(),2)


def state():
    return dict(episodes=0,queries=0,elapsed=0,valid=0,matches=0,statuses={},examples=[])


def infos(frame,keys):
    result=[]
    for slot in (0,1):
        stream=events(frame,keys)
        for row in stream:row['attacker_seat']=slot
        result.append(dict(learner_seat=slot,segment=7,native_combat_events=stream))
    return result


def test_shared_teacher_spec_is_explicit():
    validate_spec(spec())
    for key,value in [('provider','unknown'),('seed',True),('instruction_limit',0),('script','other.ai')]:
        bad=spec();bad[key]=value
        with pytest.raises(ValueError):validate_spec(bad)


def test_real_native_gate_is_bound_to_restored_source():
    from pathlib import Path
    if not Path(spec()['validation']).is_file():pytest.skip('local native evidence external')
    assert len(validated_native_source(spec()))==64


def test_student_actions_reconcile_through_terminal_and_auto_reset(monkeypatch):
    expert=make_expert(monkeypatch);stats=state()
    actions=np.array([expert.neutral,expert.neutral])
    for frame in range(3):
        obs=np.stack([packet(frame),packet(frame)])
        following=np.stack([packet(frame+1),packet(frame+1)])
        feedback=infos(frame+1,(0,)*8)
        done=np.array([frame==2]*2)
        if done.any():
            for slot in (0,1):feedback[slot]['terminal_observation']=following[slot].copy()
            following=np.stack([packet(0),packet(0)])
        before=actions.copy();expert.stage(following,feedback,done)
        labels,valid=expert.labels(obs,np.array([frame==0]*2),actions,stats)
        np.testing.assert_array_equal(actions,before)
        assert expert.pending is None
        if frame==0:assert not valid.any()
    assert all(a.ended for a in expert.actors)
    assert expert.counts['native_steps']==6 and expert.counts['terminal_closures']==2
    expert.stage(np.stack([packet(1)]*2),infos(1,(0,)*8),np.zeros(2,bool))
    expert.labels(np.stack([packet(0)]*2),np.ones(2,bool),actions,stats)
    assert stats['episodes']==4 and stats['elapsed']==8
    assert all(a.frame==0 and not a.ended for a in expert.actors)
    assert not any(a.counts['terminal_closures'] for a in expert.actors)


def test_current_label_cannot_read_the_staged_future(monkeypatch):
    first,second=make_expert(monkeypatch),make_expert(monkeypatch)
    a,b=state(),state()
    actions=np.array([first.neutral]*2)
    for expert,stats in ((first,a),(second,b)):
        expert.stage(np.stack([packet(1)]*2),infos(1,(0,)*8),np.zeros(2,bool))
        expert.labels(np.stack([packet(0)]*2),np.ones(2,bool),actions,stats)
    future=decode_shared(packet(2),schema=SCHEMA)
    future.players[1]['hp']=100
    changed=assemble_shared(public_example(future),future,public_frame=2)
    first.stage(np.stack([packet(2)]*2),infos(2,(0,)*8),np.zeros(2,bool))
    second.stage(np.stack([changed]*2),infos(2,(0,)*8),np.zeros(2,bool))
    x=first.labels(np.stack([packet(1)]*2),np.zeros(2,bool),actions,a)
    y=second.labels(np.stack([packet(1)]*2),np.zeros(2,bool),actions,b)
    for left,right in zip(x,y):np.testing.assert_array_equal(left,right)
    assert a['examples']==b['examples']


def test_missing_or_overwritten_feedback_fails_closed(monkeypatch):
    expert=make_expert(monkeypatch)
    obs=np.stack([packet(0)]*2)
    with pytest.raises(RuntimeError,match='not staged'):
        expert.labels(obs,np.ones(2,bool),np.array([expert.neutral]*2),state())
    expert.stage(obs,[{},{}],np.zeros(2,bool))
    with pytest.raises(RuntimeError,match='unconsumed'):
        expert.stage(obs,[{},{}],np.zeros(2,bool))


def test_packed_expert_storage_keeps_exact_observations_labels_and_upstream_gae():
    from stable_baselines3.common.buffers import RolloutBuffer
    space=gym.spaces.Box(-np.inf,np.inf,(8,),np.float32)
    packed=PackedExpertRolloutBuffer(3,space,gym.spaces.Discrete(90),device='cpu',n_envs=2,gamma=1.,gae_lambda=.99)
    plain=RolloutBuffer(3,space,gym.spaces.Discrete(90),device='cpu',n_envs=2,gamma=1.,gae_lambda=.99)
    packed.query=lambda *args:(np.array([3,4]),np.array([True,False]))
    snapshots=[]
    for frame in range(3):
        values=np.zeros((2,8),np.float32);values[:,0]=frame;values[:,3]=-0.
        snapshots.append(values)
        for buf in (packed,plain):
            buf.add(values,np.array([[0],[1]]),np.array([1.,-1.]),np.array([frame==0]*2),
                torch.zeros(2),torch.zeros(2))
    for buf in (packed,plain):buf.compute_returns_and_advantage(torch.zeros(2),np.ones(2,bool))
    assert isinstance(packed.observations,PackedArray)
    np.testing.assert_array_equal(packed.observations[:].view('u4'),np.stack(snapshots).view('u4'))
    np.testing.assert_array_equal(packed.advantages,plain.advantages)
    np.testing.assert_array_equal(packed.returns,plain.returns)
    samples=next(packed.get(6))
    assert samples.observations.shape==(6,8)
    assert samples.expert_mask.sum()==3


def composed():
    from pathlib import Path
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base='1.3',config_dir=str(Path('rl/config').resolve())):
        return OmegaConf.to_container(compose(config_name='train_combat_kickstarting_continuous'),resolve=True)


def test_legacy_pressure_keeps_its_public_behavior_and_identity():
    from dataclasses import replace
    from soku_rl.env.wrappers.learning import LearningInterface
    from soku_rl.policy.loader import load_policy
    shared=interface();legacy=LearningInterface(replace(shared.episode,shared_teacher_context=False),shared.config)
    config=composed()['algorithm']['opponents'][0]['policy']
    old,new=load_policy('pressure',config,legacy,'cpu'),load_policy('pressure',config,shared,'cpu')
    assert old.fingerprint==new.fingerprint
    a,b=old.spawn(17),new.spawn(17)
    for frame in range(12):
        values=packet(frame)
        assert a.act(values[:1832])==b.act(values)


def test_factory_runs_joint_update_and_resumes_without_saving_live_teacher(monkeypatch,tmp_path):
    from dataclasses import asdict
    from pathlib import Path
    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env import DummyVecEnv
    from soku_rl.env.encoding import decode_action
    from soku_rl.rl.kickstarting import KickstartingPPO
    from soku_rl.rl.ppo import create_ppo,parameter_hash
    import soku_rl.rl.continuous_expert as module
    cfg=composed();source=cfg['algorithm']['initial_policy']
    if not Path(source['path']).is_file():pytest.skip('external migrated checkpoint absent')
    monkeypatch.setattr(module,'validated_native_source',lambda spec:'SYNTHETIC_ONLY')
    config=cfg['rl'];config['timeout_payoff']='zero_at_horizon'
    config['ppo'].update(n_steps=2,batch_size=4,n_epochs=1,verbose=0)
    contract=interface()
    submitted=[]
    class Synthetic(gym.Env):
        observation_space=contract.observation_space
        action_space=contract.action_space
        def __init__(self,slot):self.slot=slot
        def reset(self,**kwargs):
            self.frame=0
            return packet(0),{}
        def step(self,action):
            self.frame+=1;submitted.append((self.slot,int(action)))
            feedback=infos(self.frame,decode_action(contract.command(int(action))).inputs)[self.slot]
            return packet(self.frame),.1,self.frame==3,False,feedback
    make_env=lambda:DummyVecEnv([lambda:Synthetic(0),lambda:Synthetic(1)])
    old_threads=torch.get_num_threads();torch.set_num_threads(2)
    model=resumed=None
    try:
        model,_=create_ppo(make_env(),contract,config,source,'cpu',17)
        before=parameter_hash(model.policy)
        assert isinstance(model.rollout_buffer.observations,PackedArray)
        model.learn(8)
        assert model.num_timesteps==8 and model._n_updates==2
        assert parameter_hash(model.policy)!=before
        assert model.kickstarting_state['elapsed']==8 and model.kickstarting_state['valid']>0
        assert model.kickstarting_state['native_guidance']['counts']['terminal_closures']==2
        assert len(submitted)==8
        path=tmp_path/'trained.zip';model.save(path)
        resumed=KickstartingPPO.load(path,device='cpu')
        assert not hasattr(resumed,'_expert')
        assert resumed.kickstarting_state['elapsed']==8
        history=tmp_path/'training.yaml'
        OmegaConf.save(OmegaConf.create(dict(episode=asdict(contract.episode),
            wrappers=asdict(contract.config),rl=config)),history)
        continued,_=create_ppo(make_env(),contract,config,
            dict(kind='checkpoint',path=str(path),training_config=str(history)),'cpu',17)
        resumed=continued
        resumed.learn(4,reset_num_timesteps=False)
        assert resumed.num_timesteps==12 and resumed.kickstarting_state['elapsed']==12
        assert resumed._expert.counts['native_steps']==4
        assert all(torch.isfinite(p).all() for p in resumed.policy.parameters())
    finally:
        if model is not None:model.get_env().close()
        if resumed is not None and resumed.get_env() is not None:resumed.get_env().close()
        torch.set_num_threads(old_threads)

from dataclasses import replace
from pathlib import Path

import numpy as np
import pickle
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from test_animation_v2 import example
from test_kickstarting import interface, spec
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.observation.animation import observe_animation,frame_width,public_history
from soku_rl.env.observation_history import ObservationHistory
from soku_rl.policy.god.query import QueryTeacher
from soku_rl.rl.animation_features import AnimationTokenFeatures
from soku_rl.rl.world_migration import expand_observation,expand_world


def test_existing_screen_core_world_positions_slots_and_visibility():
    raw,render,visibility = example()
    first,second = observe_animation(raw,render,visibility,slots=1)
    assert first.world_positions == (100,0,500,0,300,100)
    assert second.world_positions == (500,0,100,0,300,100)
    episode = EpisodeConfig(7200,8,1,0,'animation',visibility,LEGACY_MATCH,1)
    old = episode.encode(first)
    assert np.array_equal(old,np.asarray(first.values,np.float32))
    new = replace(episode,animation_world_coordinates=True)
    extended = new.encode(first)
    assert pickle.loads(pickle.dumps(first)) == first
    assert np.array_equal(extended[:len(old)],old)
    assert np.array_equal(extended[len(old):],np.asarray(first.world_positions,np.float32))
    assert len(extended) == frame_width(1,True)
    assert np.array_equal(public_history(np.tile(extended,8),1,True), public_history(np.tile(old,8),1))
    hidden = replace(render,objects=((replace(render.objects[0][0],alpha=0),),()))
    assert observe_animation(raw,hidden,visibility,slots=1)[0].world_positions[-2:] == (0.,0.)
    h = ObservationHistory(new,('player_0','player_1'))
    h.reset(0,(first,second))
    assert h.observations()['player_0'].shape == new.space().shape
    moved = replace(first,frame=1,world_positions=(120.,0.,500.,0.,300.,100.))
    h.append(1,(moved,replace(second,frame=1)))
    assert h.observations()['player_0'].reshape(8,-1)[-1,-6] == 120.
    h.reset(0,(first,second))
    assert (h.observations()['player_0'].reshape(8,-1)[:,-6] == 100.).all()
    with pytest.raises(ValueError,match='invalid animation'):
        new.encode(replace(first,world_positions=()))


def world_interface():
    contract = interface()
    contract.episode.animation_world_coordinates = True
    contract.episode.space = lambda: spaces.Box(-np.inf,np.inf,(8*frame_width(0,True),),np.float32)
    from soku_rl.env.wrappers.learning import LearningInterface
    return LearningInterface(contract.episode,contract.config)


def test_world_is_shared_and_delayed_not_teacher_only(tmp_path):
    directory = tmp_path/'lua'
    directory.mkdir()
    (directory/'fixture.ai').write_text('function main() while true do key_reset(); if x < ex then key_on(ACT_A) else key_on(ACT_B) end; yield() end end')
    contract = world_interface()
    teacher = QueryTeacher(contract,spec(directory))
    row = np.zeros(frame_width(0,True),np.float32)
    row[0],row[10],row[3],row[13],row[4] = 1,1,1,-1,1
    row[frame_width(0):] = (100,0,500,0)
    obs = np.r_[np.tile(row,8),np.zeros(8,np.float32)]
    actor = teacher.spawn(17)
    actor.query(obs)
    modified = obs.copy().reshape(-1)
    modified[:8*frame_width(0,True)].reshape(8,-1)[:,-4:] = (600,0,500,0)
    actor.query(modified)
    reply = actor.query(modified)
    assert reply.command == 257  # old shared frame, not current world positions
    assert teacher.fingerprint != QueryTeacher(interface(),spec(directory)).fingerprint


def test_world_migration_preserves_old_parameters_outputs(tmp_path):
    slots = 0
    old_space = spaces.Box(-np.inf,np.inf,(8*frame_width(slots)+8,),np.float32)
    model = PPO('MlpPolicy',None,_init_setup_model=False)
    from stable_baselines3.common.policies import ActorCriticPolicy
    model.policy_kwargs = dict(features_extractor_kwargs=dict(object_slots=slots,width=8,heads=2,features_dim=8))
    model.observation_space,model.action_space = old_space,spaces.Discrete(90)
    model.lr_schedule = lambda _:3e-4
    model.policy = ActorCriticPolicy(old_space,model.action_space,model.lr_schedule,
        net_arch=[8],features_extractor_class=AnimationTokenFeatures,**model.policy_kwargs)
    model.policy.set_training_mode(False)
    old = np.zeros((3,*old_space.shape),np.float32)
    with torch.no_grad():
        before = model.policy.get_distribution(torch.as_tensor(old)).distribution.probs.clone()
    parameters = {k:v.clone() for k,v in model.policy.state_dict().items()}
    new = expand_observation(old,slots)
    expand_world(model)
    for name,value in model.policy.state_dict().items():
        p = parameters[name]
        assert torch.equal(p,value if value.shape==p.shape else value[:,:p.shape[1]])
    with torch.no_grad():
        after = model.policy.get_distribution(torch.as_tensor(new)).distribution.probs
        torch.testing.assert_close(before,after,rtol=1e-6,atol=1e-7)
        changed = new.copy()
        changed[:,:8*frame_width(0,True)].reshape(3,8,-1)[:,:,-4:] = 800.
        torch.testing.assert_close(after,model.policy.get_distribution(torch.as_tensor(changed)).distribution.probs,rtol=0,atol=0)
    assert not model.policy.optimizer.state

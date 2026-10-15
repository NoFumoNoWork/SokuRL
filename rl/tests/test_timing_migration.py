import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from soku_rl.rl.animation_features import AnimationTokenFeatures
from soku_rl.rl.timing_migration import expand_timing


def test_expansion_preserves_old_outputs_adam_and_reload(tmp_path):
    torch.set_num_threads(4)
    env=gym.Env()
    env.observation_space=gym.spaces.Box(-np.inf,np.inf,(1544,),np.float32)
    env.action_space=gym.spaces.Discrete(90)
    model=PPO('MlpPolicy',env,n_steps=2,batch_size=2,device='cpu',seed=17,
              policy_kwargs=dict(features_extractor_class=AnimationTokenFeatures,
                                 features_extractor_kwargs=dict(width=16,heads=4)))
    x=torch.zeros((2,1544))
    extractor=model.policy.features_extractor
    loss=sum(p.square().sum() for p in model.policy.parameters())
    loss.backward();model.policy.optimizer.step();model.policy.optimizer.zero_grad()
    before_weights=[e.weight.detach().clone() for e in extractor.categories[3:]]
    moments=[model.policy.optimizer.state[e.weight]['exp_avg'].clone() for e in extractor.categories[3:]]
    with torch.no_grad():
        expected=model.policy.get_distribution(x).distribution.probs.clone()
    expand_timing(model)
    for e,weight,moment in zip(extractor.categories[3:],before_weights,moments):
        assert e.num_embeddings==8192
        torch.testing.assert_close(e.weight[:1024],weight,rtol=0,atol=0)
        state=model.policy.optimizer.state[e.weight]
        torch.testing.assert_close(state['exp_avg'][:1024],moment,rtol=0,atol=0)
        assert not state['exp_avg'][1024:].any()
    with torch.no_grad():
        torch.testing.assert_close(expected,model.policy.get_distribution(x).distribution.probs,rtol=0,atol=0)
    model.save(tmp_path/'expanded.zip')
    for device in ('cpu','cuda'):
        if device=='cuda' and not torch.cuda.is_available():continue
        loaded=PPO.load(tmp_path/'expanded.zip',device=device)
        sample=x.to(device).clone()
        sample[:, :1480].reshape(2,8,185)[...,9]=7200
        sample.requires_grad_()
        value=loaded.policy.features_extractor(sample)
        assert torch.isfinite(value).all()
        value.square().mean().backward()
        loaded.policy.optimizer.step();loaded.policy.optimizer.zero_grad()
        for invalid in (-1,8192):
            bad=sample.detach().clone()
            bad[:,9]=invalid
            with pytest.raises(ValueError,match='category'):
                loaded.policy.features_extractor(bad)


def test_default_old_capacity_remains_strict():
    space=gym.spaces.Box(-np.inf,np.inf,(1544,),np.float32)
    feature=AnimationTokenFeatures(space)
    x=torch.zeros((1,1544));x[:,9]=1024
    with pytest.raises(ValueError,match='category 5'):feature(x)


def test_migration_checks_cached_artifact_integrity(tmp_path):
    from soku_rl.rl.timing_migration import migrate_checkpoint
    from omegaconf import OmegaConf
    torch.set_num_threads(4)
    env=gym.Env()
    env.observation_space=gym.spaces.Box(-np.inf,np.inf,(1544,),np.float32)
    env.action_space=gym.spaces.Discrete(90)
    model=PPO('MlpPolicy',env,n_steps=2,batch_size=2,device='cpu',
              policy_kwargs=dict(features_extractor_class=AnimationTokenFeatures))
    source=tmp_path/'source.zip';model.save(source)
    config=tmp_path/'config.yaml'
    OmegaConf.save(OmegaConf.create({'rl':{'ppo':{'policy_kwargs':{'features_extractor_kwargs':{}}}}}),config)
    destination=tmp_path/'migration'
    files=migrate_checkpoint(source,config,destination)
    assert files==migrate_checkpoint(source,config,destination)
    files[0].write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash mismatch'):
        migrate_checkpoint(source,config,destination)

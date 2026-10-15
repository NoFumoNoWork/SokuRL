import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from soku_rl.rl.ppo import initialize_ppo
from soku_rl.rl.masked_policy import CardMaskedActorCriticPolicy
from test_policy_artifacts import interface


def test_actual_ppo_mask_sampling_gradients_and_checkpoint(tmp_path):
    contract=interface()
    class TinyEnv(gym.Env):
        observation_space=contract.observation_space
        action_space=contract.action_space
        def reset(self,seed=None,options=None):
            super().reset(seed=seed)
            return np.zeros(self.observation_space.shape,np.float32),{}
        def step(self,action):
            assert not contract.commands[int(action)]&48
            return np.zeros(self.observation_space.shape,np.float32),0.,False,False,{}
    config=dict(name='ppo',policy_type='mlp',timeout_payoff='zero_at_horizon',ppo=dict(
        mask_card_actions=True,n_steps=4,batch_size=4,n_epochs=1,gamma=1.,policy_kwargs=dict(net_arch=[8])))
    model,_=initialize_ppo(PPO,'MlpPolicy',TinyEnv(),contract,config,{'kind':'fresh'},'cpu',17)
    assert isinstance(model.policy,CardMaskedActorCriticPolicy)
    assert model.action_space.n==90
    blocked=[i for i,c in enumerate(contract.commands) if c&48]
    assert len(blocked)==18
    values=torch.zeros((3,*contract.observation_space.shape))
    probs=model.policy.get_distribution(values).distribution.probs
    assert torch.all(probs[:,blocked]==0)
    assert torch.allclose(probs.sum(-1),torch.ones(3))
    assert not any(contract.commands[int(i)]&48 for i in torch.multinomial(probs[0],10000,replacement=True))
    model.learn(8)
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    path=tmp_path/'masked.zip';model.save(path)
    loaded=PPO.load(path,device='cpu')
    assert isinstance(loaded.policy,CardMaskedActorCriticPolicy)
    assert torch.all(loaded.policy.get_distribution(values).distribution.probs[:,blocked]==0)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.policy.parameters())


def test_old_default_ppo_remains_unmasked():
    contract=interface()
    env=gym.Env();env.observation_space=contract.observation_space;env.action_space=contract.action_space
    config=dict(name='ppo',policy_type='mlp',timeout_payoff='zero_at_horizon',ppo=dict(
        n_steps=2,batch_size=2,gamma=1.,policy_kwargs=dict(net_arch=[8])))
    model,_=initialize_ppo(PPO,'MlpPolicy',env,contract,config,{'kind':'fresh'},'cpu',17)
    assert not isinstance(model.policy,CardMaskedActorCriticPolicy)
    assert torch.all(model.policy.get_distribution(torch.zeros((1,*contract.observation_space.shape))).distribution.probs>0)

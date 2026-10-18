"""Explicit OFFLINE current-state expansion; no native reads or training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch

from soku_rl.env.observation.shared_native import (PUBLIC_FEATURES,SHARED_FEATURES,SharedNativeContract)
from soku_rl.rl.shared_native_features import SharedNativeFeatures,initialize_shared_policy
from soku_rl.rl.ppo import parameter_hash


class ShapeOnlyEnv(gym.Env):
    def __init__(self):
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(SHARED_FEATURES,),np.float32)
        self.action_space=gym.spaces.Discrete(90)

    def reset(self,**kwargs):
        raise RuntimeError('offline migration has no native environment')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    source_path=Path(args.checkpoint).resolve(strict=True)
    output=Path(args.output).resolve()
    if (not source_path.is_relative_to(root) or not output.is_relative_to(root/'rl/runs')
            or output.exists()):
        raise ValueError('use existing repository checkpoint and NEW local run output')
    output.mkdir(parents=True)
    checkpoint_hash=hashlib.sha256(source_path.read_bytes()).hexdigest()
    source=PPO.load(source_path,device='cpu')
    source.policy.set_training_mode(False)
    original_hash=parameter_hash(source.policy)
    kwargs=copy.deepcopy(source.policy_kwargs)
    public_kwargs=kwargs.pop('features_extractor_kwargs')
    kwargs['features_extractor_class']=SharedNativeFeatures
    kwargs['features_extractor_kwargs']=dict(public_kwargs=public_kwargs,
        features_dim=source.policy.features_extractor.features_dim)
    # No continuation claim: new optimizer and no environment stepping. The
    # original Actor/Critic weights and action mask are strictly transferred.
    target=PPO(type(source.policy),ShapeOnlyEnv(),device='cpu',seed=17,
        n_steps=2,batch_size=2,policy_kwargs=kwargs)
    metadata=initialize_shared_policy(target.policy,source.policy)
    target.policy.set_training_mode(False)
    # Finite contract probes only, NOT captured native/game-legal snapshots.
    old=np.zeros((3,PUBLIC_FEATURES),np.float32)
    new=np.zeros((3,SHARED_FEATURES),np.float32)
    for index in range(3):
        old[index,0]=old[index,10]=1
        old[index,3]=1;old[index,13]=-1
        old[index,4]=1;old[index,5]=index
    new[:,:PUBLIC_FEATURES]=old
    new[:,PUBLIC_FEATURES+1]=np.arange(3,dtype=np.float32)/65536.
    with torch.no_grad():
        a=source.policy.get_distribution(torch.as_tensor(old)).distribution.probs
        b=target.policy.get_distribution(torch.as_tensor(new)).distribution.probs
        av=source.policy.predict_values(torch.as_tensor(old))
        bv=target.policy.predict_values(torch.as_tensor(new))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(av,bv,rtol=0,atol=0)
    if parameter_hash(source.policy)!=original_hash:
        raise RuntimeError('original checkpoint policy changed')
    target.shared_native_state=dict(schema='shared_current_native_v1',training_ready=False,
        purpose='offline policy-weight expansion only; NOT a resumable training configuration',
        control='UNASSIGNED',native_capture_verified=False)
    target.save(output/'model.zip')
    loaded=PPO.load(output/'model.zip',device='cpu')
    loaded.policy.set_training_mode(False)
    with torch.no_grad():
        torch.testing.assert_close(b,loaded.policy.get_distribution(torch.as_tensor(new)).distribution.probs,rtol=0,atol=0)
    report=dict(status='PASS_OFFLINE_SHARED_OBSERVATION_MIGRATION',contract=SharedNativeContract().report(),
        source=str(source_path),source_sha256=checkpoint_hash,source_parameters=original_hash,
        source_steps=source.num_timesteps,new_steps=target.num_timesteps,
        max_probability_error=float((a-b).abs().max()),max_value_error=float((av-bv).abs().max()),
        initialized_parameters=parameter_hash(target.policy),transfer=metadata,
        source_unchanged=hashlib.sha256(source_path.read_bytes()).hexdigest()==checkpoint_hash,
        native_steps=0,gradient_updates=0,control_assessment='unresolved; do not attach this to expert PPO',
        files={'model.zip':hashlib.sha256((output/'model.zip').read_bytes()).hexdigest()})
    (output/'migration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','source_steps','new_steps','max_probability_error','max_value_error','native_steps','gradient_updates')}))


if __name__=='__main__':main()

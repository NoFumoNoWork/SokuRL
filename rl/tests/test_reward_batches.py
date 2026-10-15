import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO

from soku_rl.rl.reward_batches import RewardBatch,append_batch,plot_batches
from soku_rl.rl.training import EpisodeRecords
from soku_rl.rl.curriculum import FixedOpponentSchedule


def parts(**values):
    return dict(damage=0.,combo=0.,counter=0.,passive=0.,terminal=0.)|values


def test_batches_cover_unfinished_episodes_and_separate_terminal_signs(tmp_path):
    batch=RewardBatch(10000.,100)
    batch.add(parts(combo=40,passive=-10),.003)
    batch.add(parts(terminal=10000),1.,'win')
    batch.add(parts(terminal=-10000),-1.,'loss')
    row=batch.row(1,103,True)
    assert row['hp']['terminal']==0
    assert row['hp']['win']==10000 and row['hp']['loss']==-10000
    assert row['hp']['combo']==40 and row['hp']['passive']==-10
    assert row['outcome_counts']['win']==row['outcome_counts']['loss']==1
    assert abs(row['reward_residual'])<1e-12
    append_batch(tmp_path,row);plot_batches(tmp_path)
    assert len((tmp_path/'reward_batches.csv').read_text().splitlines())==2
    assert ET.parse(tmp_path/'reward_batches.svg').getroot().tag.endswith('svg')
    fresh=RewardBatch(10000.,103);fresh.add(parts(),0.)
    assert fresh.row(2,104,False)['hp']['combo']==0
    with pytest.raises(ValueError,match='coverage'):fresh.row(2,105,False)


@pytest.mark.parametrize('start',[0,4])
def test_actual_ppo_callback_records_each_rollout_without_completed_episode(tmp_path,start):
    import torch
    torch.set_num_threads(4)
    class Env(gym.Env):
        observation_space=gym.spaces.Box(-1,1,(2,),np.float32)
        action_space=gym.spaces.Discrete(2)
        def reset(self,seed=None,options=None):
            super().reset(seed=seed)
            return np.zeros(2,np.float32),{}
        def step(self,action):
            return np.zeros(2,np.float32),.001,False,False,dict(
                combat_reward_parts=parts(combo=15,passive=-5))
    model=PPO('MlpPolicy',Env(),n_steps=4,batch_size=4,n_epochs=1,device='cpu',seed=17)
    if start:model.learn(start)
    callback=EpisodeRecords(tmp_path,4096,FixedOpponentSchedule(),reward_scale=10000.)
    model.learn(8,callback=callback,reset_num_timesteps=False)
    rows=[json.loads(line) for line in (tmp_path/'reward_batches.jsonl').read_text().splitlines()]
    assert len(rows)==2 and [r['transitions'] for r in rows]==[4,4]
    assert rows[0]['start_steps']==start and rows[1]['start_steps']==start+4
    assert all(r['hp']['combo']==60 and r['hp']['passive']==-20 for r in rows)
    assert all(r['hp']['terminal']==0 and abs(r['reward_residual'])<1e-8 for r in rows)
    assert not callback.records
    assert (tmp_path/'reward_batches.svg').is_file()


def test_partial_batch_and_missing_terminal_evidence(tmp_path):
    batch=RewardBatch(10000.,12)
    with pytest.raises(ValueError,match='without terminal'):batch.add(parts(terminal=1),.0001)
    batch.add(parts(passive=-20),-.002)
    row=batch.row(1,13,False);append_batch(tmp_path,row)
    assert row['complete'] is False

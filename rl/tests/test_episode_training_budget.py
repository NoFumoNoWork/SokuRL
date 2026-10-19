"""Episode-limited PPO stops through the upstream callback contract."""
import json
import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from soku_rl.rl.training import EpisodeRecords
from soku_rl.rl.curriculum import FixedOpponentSchedule


class ShortEpisodes(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(2,),np.float32)
    action_space=gym.spaces.Discrete(2)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed);self.frame=0
        return np.zeros(2,np.float32),{}

    def step(self,action):
        self.frame+=1
        done=self.frame==3
        return np.zeros(2,np.float32),float(done),done,False,dict(
            frame=self.frame,decision_frames=1,latency_frames=0,outcome='p1_win' if done else 'ongoing',
            source_truncated=False,training_context={'player':0,'opponent':'synthetic'},
            combat_reward_parts=dict(damage=0.,combo=0.,counter=0.,passive=0.,terminal=10000. if done else 0.))


def test_episode_stop_keeps_final_partial_batch_and_record(tmp_path):
    model=PPO('MlpPolicy',ShortEpisodes(),n_steps=4,batch_size=4,n_epochs=1,device='cpu',seed=1)
    callback=EpisodeRecords(tmp_path,1000,FixedOpponentSchedule(),10000.,max_episodes=3)
    model.learn(100,callback=callback)
    assert model.num_timesteps==9 and model._n_updates==2
    assert len(callback.records)==3
    progress=json.loads((tmp_path/'progress.json').read_text())
    assert len(progress['episodes'])==3
    rows=[json.loads(line) for line in (tmp_path/'reward_batches.jsonl').read_text().splitlines()]
    assert [r['transitions'] for r in rows]==[4,4,1]
    assert rows[-1]['complete'] is False
    assert sum(r['outcome_counts']['win'] for r in rows)==3


@pytest.mark.parametrize('value',[0,-1,True,2.5])
def test_invalid_episode_budget(tmp_path,value):
    with pytest.raises(ValueError):
        EpisodeRecords(tmp_path,1000,FixedOpponentSchedule(),max_episodes=value)


def test_simultaneous_endings_are_reported_not_discarded(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    env=DummyVecEnv([lambda:Monitor(ShortEpisodes()),lambda:Monitor(ShortEpisodes())])
    try:
        model=PPO('MlpPolicy',env,n_steps=4,batch_size=4,n_epochs=1,device='cpu',seed=1)
        callback=EpisodeRecords(tmp_path,1000,FixedOpponentSchedule(),10000.,max_episodes=3)
        model.learn(100,callback=callback)
        assert model.num_timesteps==12
        assert len(callback.records)==4
        assert len(json.loads((tmp_path/'progress.json').read_text())['episodes'])==4
    finally:
        env.close()

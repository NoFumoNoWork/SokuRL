"""All-neutral native V2 episodes versus an independent passive-cost formula."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import gymnasium as gym
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.encoding import AGENTS, encode_action
from soku_rl.env.vector_env import TwoPlayerVectorEnv
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.rl.ppo import create_ppo


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    root=Path(os.environ['SOKURL_RL_ROOT'])
    with initialize_config_dir(version_base='1.3',config_dir=str(root/'config')):
        config=OmegaConf.to_container(compose(config_name='train_combat_v2'),resolve=True)
    args.output.with_suffix('.config.yaml').write_text(OmegaConf.to_yaml(OmegaConf.create(config)),encoding='utf-8')
    episode=EpisodeConfig.from_dict(config['episode'])
    interface=LearningInterface(episode,LearningConfig(**config['wrappers']))
    dummy=gym.Env()
    dummy.observation_space,dummy.action_space=interface.observation_space,interface.action_space
    model,_=create_ppo(dummy,interface,config['algorithm'],{'kind':'fresh'},config['device'],config['seed'])
    with torch.no_grad():
        distribution=model.policy.get_distribution(torch.zeros((4,*interface.observation_space.shape),device=model.device)).distribution
        probabilities=distribution.probs
        blocked=[i for i,c in enumerate(interface.commands) if c&48]
        if len(blocked)!=18 or torch.any(probabilities[:,blocked]!=0):raise AssertionError('card mask failed')
        if not torch.allclose(probabilities.sum(-1),torch.ones(4,device=model.device)):raise AssertionError('invalid distribution')
        if not torch.isfinite(distribution.entropy()).all():raise AssertionError('invalid masked entropy')
    del model
    backend=WorkerBackend(log_path=str(args.output.with_suffix('.worker.log')),**config['runtime'])
    backend.configure_observation(episode.backend_observation())
    env=TwoPlayerVectorEnv(backend,config['num_envs'],episode)
    neutral=encode_action((0,)*8)
    slots=list(range(config['num_envs']))
    seeds={s:config['seed']+s for s in slots}
    parameters=env.episodes[0].reward.config
    def expected_cost(t):
        return min(parameters.passive_cap,10000*parameters.passive_b*
                   max(0,(t-parameters.passive_grace)/parameters.passive_grace)**parameters.passive_exponent)
    report=dict(status='RUNNING',config='train_combat_v2',instances=len(slots),
                policy_mask=dict(actions=90,blocked=blocked,allowed=72,blocked_probability=0.),
                samples=[],first_penalty={},returns={},reset_verified=False)
    start=time.perf_counter()
    try:
        env.reset(seeds)
        pids={str(s):v['pid'] for s,v in backend.request('status',slots).items()}
        totals={s:[0.,0.] for s in slots}
        sample_frames={1,120,121,240,360,544,545,546,episode.max_frames}
        with args.output.with_suffix('.frames.csv').open('x',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream)
            writer.writerow(('frame','slot','seat','reward','passive_hp','expected_passive_hp','terminal_hp','passive_total_hp'))
            for frame in range(1,episode.max_frames+1):
                _,rewards,terms,truncs,infos=env.step({s:dict.fromkeys(AGENTS,neutral) for s in slots})
                increment=expected_cost(frame)-expected_cost(frame-1)
                terminal=parameters.terminal_timeout if frame==episode.max_frames else 0.
                for s in slots:
                    ledger=env.episodes[s].reward
                    if ledger.hp!=(10000,10000):raise AssertionError('neutral players took damage')
                    if ledger.passive!=[frame,frame]:raise AssertionError(f'idle eligibility differs at slot{s}/frame{frame}')
                    for seat,agent in enumerate(AGENTS):
                        part=infos[s][agent]['combat_reward_parts']
                        if any(part[k]!=0 for k in ('damage','combo','counter')):raise AssertionError('unexpected combat reward')
                        if not math.isclose(part['passive'],-increment,abs_tol=1e-8):raise AssertionError('passive increment mismatch')
                        if part['terminal']!=terminal:raise AssertionError('terminal reward mismatch')
                        value=(-increment+terminal)/parameters.scale
                        if not math.isclose(rewards[s][agent],value,abs_tol=1e-10):raise AssertionError('reward mismatch')
                        totals[s][seat]+=rewards[s][agent]
                        key=f'{s}:{seat}'
                        if part['passive']<0 and key not in report['first_penalty']:report['first_penalty'][key]=frame
                        writer.writerow((frame,s,seat,rewards[s][agent],part['passive'],-increment,terminal,ledger.passive_spent[seat]))
                    if frame in sample_frames:
                        report['samples'].append(dict(slot=s,frame=frame,parts=[infos[s][a]['combat_reward_parts'] for a in AGENTS],
                            spent=ledger.passive_spent.copy(),hp=ledger.hp))
                    if any(terms[s].values()) or any(truncs[s].values())!=(frame==episode.max_frames):
                        raise AssertionError('unexpected episode end')
        report['returns']=totals
        report['pids']=pids
        env.reset({s:seeds[s]+100 for s in slots})
        if any(env.episodes[s].reward.passive_spent!=[0.,0.] for s in slots):raise AssertionError('reset budget not cleared')
        for frame in range(1,122):
            env.step({s:dict.fromkeys(AGENTS,neutral) for s in slots})
        if {str(s):v['pid'] for s,v in backend.request('status',slots).items()}!=pids:raise AssertionError('reset replaced process')
        if any(not math.isclose(env.episodes[s].reward.passive_spent[seat],expected_cost(121),abs_tol=1e-8)
               for s in slots for seat in (0,1)):raise AssertionError('reset passive trajectory mismatch')
        report.update(status='PASS',frames=episode.max_frames,reset_verified=True,reset_frames=121,
                      expected_return=(-parameters.passive_cap+parameters.terminal_timeout)/parameters.scale)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        env.close()
        report['seconds']=time.perf_counter()-start
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))


if __name__=='__main__':main()

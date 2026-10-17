"""Bounded native wiring test,not training or a battle-mechanics measurement."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from sokurl_platform.command_input import CommandPlayer,compile_commands
from soku_rl.env.encoding import AGENTS,encode_action
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import MatchConfig,PlayerSetup
from soku_rl.env.vector_env import TwoPlayerVectorEnv
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.wrappers.learning import LearningConfig,LearningVectorEnv
from soku_rl.policy.hierarchical_bc import HierarchicalEpisode
from soku_rl.policy.hierarchical_control import candidate_opportunity
from soku_rl.policy.loader import load_policy
from soku_rl.policy.matchups import opponent_interface
from soku_rl.replay.hierarchical import LATEST
from soku_rl.replay.human_dataset import sha256,save_json
from soku_rl.rl.factorized_bc import load_event_actor as load_hierarchical
from compare_v3c import NativeStates


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=Path('rl/config/hierarchical_native_smoke.yaml'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2];out=args.output.resolve()
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    if args.checkpoint is not None:cfg['checkpoint']=str(args.checkpoint.resolve(strict=True))
    if out.exists() or not out.is_relative_to(root/'rl/runs'):raise ValueError('new project-local output required')
    if cfg['legality_mode']!='masked_native_feedback' or (cfg['warmup_frames'],cfg['fixture_frames'],cfg['model_frames'],cfg['reset_model_frames'])!=(600,150,256,64):
        raise ValueError('bounded opt-in smoke contract differs')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    config=OmegaConf.to_container(OmegaConf.load(cfg['source_config']),resolve=True)
    actor,metadata=load_hierarchical(cfg['checkpoint'],'cpu');catalog=metadata['catalog']
    report=dict(status='RUNNING',config=cfg,checkpoint_sha256=sha256(cfg['checkpoint']),
        model_training='NOT RUN',strength_evaluation='NOT RUN',legality='CANDIDATES_NOT_ENGINE_LEGALITY',
        fixtures=[],episodes=[],events=[],native_frames=[0,0],event_counts={},input_examples=[],reset_checks=[])
    neutral=(0,)*8;frame_records=[];started=time.perf_counter()
    try:
        with ExitStack() as stack:
            backend=WorkerBackend([sys.executable,'-B',str(root/'rl/tools/rollout_worker.py')],str(root),
                str(out/'worker.log'),120,45,True,str(root/'th123_jp'),fresh_reset=True,
                cards_disabled=True,fixed_weather='clear',combat_events=True,native_input_counters=True)
            stack.callback(backend.close)
            episode=EpisodeConfig.from_dict(config['episode'])
            backend.configure_observation(episode.backend_observation())
            env=LearningVectorEnv(TwoPlayerVectorEnv(backend,2,episode),LearningConfig(**config['wrappers']))
            stack.callback(env.close)
            reader=NativeStates(backend,stack)
            matches={0:MatchConfig(PlayerSetup(1,0,0),PlayerSetup(0,0,0)),
                     1:MatchConfig(PlayerSetup(0,0,0),PlayerSetup(1,0,0))}
            entry=next(e for e in config['algorithm']['opponents'] if e['name']=='pressure')
            pressure=load_policy('pressure',entry['policy'],opponent_interface(env.interface,
                config['algorithm']['matchups']['learner'],entry),'cpu')
            neutral_id=env.interface.inverse[encode_action(neutral)]
            for round_index in (0,1):
                seeds=cfg['seeds'] if round_index==0 else cfg['reset_seeds']
                observations,_=env.reset_matchups(dict(enumerate(seeds)),matches)
                pids={s:info['pid'] for s,info in backend.request('status',[0,1]).items()}
                if round_index and pids!=report['reset_checks'][0]['pids']:
                    raise RuntimeError('fast reset unexpectedly replaced a game process')
                for s in (0,1):
                    if int(observations[s][AGENTS[s]][1299])!=1:
                        raise RuntimeError('wrong learner character/seat in public observation')
                controllers={s:HierarchicalEpisode(actor,catalog,env.interface,'cpu',cfg['max_wait'],cfg['legality_mode']) for s in (0,1)}
                if any(env.episodes[s].frame!=0 or controllers[s].history.frame is not None
                       or controllers[s].decisions for s in (0,1)):
                    raise RuntimeError('reset leaked frame or controller history')
                report['reset_checks'].append(dict(episode=round_index,pids=pids,
                    frame_zero=True,new_controller_history=True,same_process_fast_reset=bool(round_index)))
                opponents={s:pressure.spawn(seeds[s]+7) for s in (0,1)}
                events={s:[] for s in (0,1)};local=0
                def advance(actions,phase):
                    nonlocal observations,events,local
                    observations,rewards,terms,truncs,infos=env.step(actions);local+=1
                    if any(any(t.values()) for group in (terms,truncs) for t in group.values()):
                        raise RuntimeError('unexpected terminal in bounded wiring test')
                    for s in (0,1):
                        report['native_frames'][s]+=1
                        if env.episodes[s].frame!=local:raise RuntimeError('lost/repeated native frame')
                        events[s]=infos[s][AGENTS[s]]['native_combat_events']
                        for event in events[s]:
                            key=f"{s}:{event['kind']}"
                            report['event_counts'][key]=report['event_counts'].get(key,0)+1
                            if phase=='known5B' and len(report['input_examples'])<12 and event['kind']=='NATIVE_INPUT_SAMPLE':
                                report['input_examples'].append(dict(slot=s,**event))
                        report['events'].extend(dict(slot=s,episode=round_index,**e) for e in events[s]
                            if e['kind']=='ACTION_TRANSITION' and e['attacker_seat']==s)
                    return infos
                def remember_before_step():
                    for s in (0,1):
                        value=observations[s][AGENTS[s]]
                        controllers[s].history.observe(value[LATEST],local,None if local==0 else value[-8:])
                for _ in range(cfg['warmup_frames']):
                    remember_before_step()
                    advance({s:{a:neutral_id for a in AGENTS} for s in (0,1)},'warmup')
                if round_index==0:
                    fixture={s:CommandPlayer(compile_commands('5B')) for s in (0,1)}
                    accepted={s:False for s in (0,1)}
                    for _ in range(cfg['fixture_frames']):
                        remember_before_step();actions={}
                        for s in (0,1):
                            state=reader(s,s);keys=fixture[s].next(local,state['facing'],segment=state['segment'])
                            actions[s]={AGENTS[s]:env.interface.inverse[encode_action(keys)],AGENTS[1-s]:neutral_id}
                        advance(actions,'known5B')
                        for s in (0,1):
                            fixture[s].observe_native(events[s],s)
                            accepted[s]|=any(e['kind']=='ACTION_TRANSITION' and e['attacker_seat']==s
                                and e.get('accepted') and e.get('action_after')==400 for e in events[s])
                    report['fixtures'].append(dict(name='EXISTING_MARISA_5B_ACCEPTANCE',accepted=accepted))
                    if not all(accepted.values()):raise RuntimeError('previously verified5B fixture did not reproduce')
                count=cfg['model_frames'] if round_index==0 else cfg['reset_model_frames']
                for _ in range(count):
                    joint={}
                    for s in (0,1):
                        state=reader(s,s);op=candidate_opportunity(catalog,state,cfg['allowed_candidates'])
                        action,feedback=controllers[s].act(observations[s][AGENTS[s]],frame=local,
                            facing=state['facing'],segment=state['segment'],events=events[s],seat=s,opportunity=op)
                        joint[s]={AGENTS[s]:action,AGENTS[1-s]:opponents[s].act(observations[s][AGENTS[1-s]])}
                        frame_records.append(dict(episode=round_index,slot=s,frame=local,primitive=action,**feedback))
                    advance(joint,'model')
                for s in (0,1):
                    controllers[s].controller.observe_feedback(events[s],s)
                    controllers[s].controller.finish_attempt('SMOKE_END_CENSORED')
                    report['episodes'].append(dict(episode=round_index,slot=s,learner_seat=s,
                        world_seed=seeds[s],decisions=controllers[s].decisions,evaluations=controllers[s].evaluations,
                        event_history=len(controllers[s].history.events),input_history=len(controllers[s].history.inputs),
                        attempts=controllers[s].controller.attempts,frame=local))
            if not all(r['decisions']>0 for r in report['episodes']):raise RuntimeError('one slot never made a model decision')
            for row in frame_records:
                if row['game_legality_known']:raise RuntimeError('candidate falsely claimed engine legality')
            report.update(status='PASS_BOUNDED_NATIVE_WIRING',scope='Knownfixture plus shortmodel-loop,not policy strength/all-command legality')
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        report['seconds']=time.perf_counter()-started
        save_json(out/'frames.json',frame_records);save_json(out/'result.json',report)
    print(json.dumps(dict(status=report['status'],native_frames=report['native_frames'],
        fixtures=report['fixtures'],seconds=report['seconds'],
        episodes=[{k:v for k,v in e.items() if k!='attempts'} for e in report['episodes']]),indent=2))


if __name__=='__main__':main()

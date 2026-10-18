"""Bounded student-only native teacher sidecar; no optimizer or game-rule probe."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
import torch

from game_runtime.identity import fingerprints
from soku_rl.env.encoding import AGENTS,decode_action,encode_action
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import MatchConfig,PlayerSetup
from soku_rl.env.vector_env import TwoPlayerVectorEnv
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.wrappers.learning import LearningConfig,LearningInterface,LearningVectorEnv
from soku_rl.env.observation.shared_native import SCHEMA,decode_shared
from soku_rl.policy.loader import load_policy
from soku_rl.policy.god.package import ScriptPackage
from soku_rl.policy.god.continuous import ContinuousSourceTeacher
from soku_rl.policy.god.native_feedback import NativeTeacherFeedback
from soku_rl.rl.ppo import parameter_hash
from soku_rl.replay.human_dataset import sha256


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2];out=args.output.resolve()
    if out.exists() or not out.is_relative_to(root/'rl/runs'):
        raise ValueError('NEW project-local output required')
    out.mkdir(parents=True)
    def save(name,data):
        (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    report=dict(status='RUNNING',native_frames=[0,0],episodes=[],training_steps=0,
        student_only=True,teacher_game_writes=0,move_legality='NOT_A_COMPLETE_ENGINE_LEGAL_TABLE')
    started=time.perf_counter();before=fingerprints();save('before.json',before)
    checkpoint=root/'rl/runs/teacher-shared-native-20261018/student-v1/model.zip'
    cfg=OmegaConf.to_container(OmegaConf.load(root/'rl/runs/combat-v2-adaptive-20261015/campaign/train-4/config.yaml'),resolve=True)
    episode=replace(EpisodeConfig.from_dict(cfg['episode']),animation_world_coordinates=True,
        shared_teacher_context=True)
    learning=LearningConfig(**cfg['wrappers'])
    torch.set_num_threads(4);torch.manual_seed(71303100)
    model=PPO.load(checkpoint,device='cpu');model.policy.set_training_mode(False)
    if model.observation_space.shape!=(592658,) or model.action_space.n!=90:
        raise ValueError('explicit shared-state/90-action migrated model required')
    initial=parameter_hash(model.policy)
    report.update(checkpoint_sha256=sha256(checkpoint),initial_parameter_hash=initial,
        config=dict(episode=asdict(episode),learning=asdict(learning)))
    package=ScriptPackage(root/'third_party/th123_ai/package/th123ai/script',
        root/'third_party/th123_ai/source/th123_ai/api.ai')
    try:
        with ExitStack() as stack:
            backend=WorkerBackend([sys.executable,'-B',str(root/'rl/tools/rollout_worker.py')],str(root),
                str(out/'worker.log'),120,45,True,str(root/'th123_jp'),fresh_reset=True,
                cards_disabled=True,fixed_weather='clear',combat_events=True,native_input_counters=True)
            stack.callback(backend.close)
            backend.configure_observation(episode.backend_observation())
            env=LearningVectorEnv(TwoPlayerVectorEnv(backend,2,episode),learning)
            stack.callback(env.close)
            pressure_entry=next(e for e in cfg['algorithm']['opponents'] if e['name']=='pressure')
            opponent_interface=LearningInterface(replace(episode,shared_teacher_context=False),learning)
            pressure=load_policy('pressure',pressure_entry['policy'],opponent_interface,'cpu')
            matches={0:MatchConfig(PlayerSetup(1,0,0),PlayerSetup(0,0,0)),
                1:MatchConfig(PlayerSetup(0,0,0),PlayerSetup(1,0,0))}
            neutral=env.interface.action(encode_action((0,)*8))
            mask=tuple(not any(decode_action(env.interface.command(i)).inputs[-2:]) for i in range(90))
            pids_before=None
            for round_index,frames in enumerate((128,64)):
                seeds={0:71303101+round_index*10,1:71303102+round_index*10}
                observations,infos=env.reset_matchups(seeds,matches)
                pids={s:info['pid'] for s,info in backend.request('status',[0,1]).items()}
                if round_index and pids!=pids_before:
                    raise RuntimeError('fast Reset replaced game instances')
                pids_before=pids
                actors={s:ContinuousSourceTeacher(package,'01_marisa_main.ai',seeds[s]+3,
                    env.interface,initial_action=neutral) for s in (0,1)}
                joins={s:NativeTeacherFeedback(env.interface,s,infos[s][AGENTS[s]]['segment'],
                    initial_facing=int(decode_shared(observations[s][AGENTS[s]],schema=SCHEMA).players[0]['dir'])) for s in (0,1)}
                probes={s:NativeTeacherFeedback(env.interface,s,infos[s][AGENTS[s]]['segment'],
                    initial_facing=joins[s].previous_facing) for s in (0,1)}
                opponents={s:pressure.spawn(seeds[s]+7) for s in (0,1)}
                previous={s:None for s in (0,1)};boundaries={s:False for s in (0,1)}
                evidence={s:None for s in (0,1)}
                rows=[]
                for frame in range(frames):
                    actions={}
                    for s in (0,1):
                        value=observations[s][AGENTS[s]]
                        native=decode_shared(value,schema=SCHEMA)
                        if native.players[0]['char']!=1 or native.players[1]['char']!=0:
                            raise RuntimeError('wrong learner role/seat')
                        advice=actors[s].query(value,schema=SCHEMA,receipt=previous[s],
                            trusted_boundary=boundaries[s],legal_actions=mask)
                        with torch.no_grad():
                            student=int(model.predict(value,deterministic=False)[0])
                        if not mask[student]:raise RuntimeError('student selected masked card input')
                        opponent=int(opponents[s].act(observations[s][AGENTS[1-s]][:1832]))
                        actions[s]={AGENTS[s]:student,AGENTS[1-s]:opponent}
                        rows.append(dict(round=round_index,slot=s,frame=frame,advice=advice,
                            student=student,agreement=advice['training_label']==student,
                            prior_native_evidence=evidence[s]))
                    observations,rewards,terms,truncs,infos=env.step(actions)
                    if any(any(t.values()) for group in (terms,truncs) for t in group.values()):
                        raise RuntimeError('unexpected terminal in bounded wiring test')
                    for s in (0,1):
                        info=infos[s][AGENTS[s]]
                        previous[s],boundaries[s],evidence[s]=joins[s].observe(frame+1,
                            info['native_combat_events'],actions[s][AGENTS[s]],actors[s],
                            decode_shared(observations[s][AGENTS[s]],schema=SCHEMA))
                        # Independent acceptance diagnostic for the student's
                        # ACTUAL sampled input,not a claim it adopted the teacher.
                        probe=SimpleNamespace(pending={'candidate':actions[s][AGENTS[s]]},serial=frame+1)
                        _,_,probe_evidence=probes[s].observe(frame+1,info['native_combat_events'],
                            actions[s][AGENTS[s]],probe,decode_shared(observations[s][AGENTS[s]],schema=SCHEMA))
                        rows[-2+s]['student_acceptance_diagnostic']=probe_evidence
                        report['native_frames'][s]+=1
                save(f'round-{round_index}-trace.json',rows)
                report['episodes'].append(dict(round=round_index,pids=pids,reset_frame_zero=True,
                    same_process_reset=bool(round_index),seeds=seeds,
                    teachers={s:actors[s].report() for s in (0,1)},
                    joins={s:dict(joins[s].counts) for s in (0,1)},
                    independent_student_acceptance={s:dict(probes[s].counts) for s in (0,1)}))
                save('report.json',report)
        report['final_parameter_hash']=parameter_hash(model.policy)
        if report['final_parameter_hash']!=initial:raise RuntimeError('model changed')
        after=fingerprints();save('after.json',after)
        if after!=before:raise RuntimeError('runtime source/game restoration mismatch')
        labels=sum(e['teachers'][s]['counts'].get('labels',0) for e in report['episodes'] for s in (0,1))
        recoveries=sum(e['teachers'][s]['counts'].get('recoveries',0) for e in report['episodes'] for s in (0,1))
        report.update(status='PASS_BOUNDED_NATIVE_SIDECAR' if labels and recoveries else 'BLOCKED_INSUFFICIENT_GUIDANCE',
            labels=labels,recoveries=recoveries)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        report['seconds']=time.perf_counter()-started
        save('report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

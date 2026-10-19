"""Bounded dual-stream trainer over unchanged Platform observations/execution."""
import argparse
from contextlib import contextmanager,ExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from stable_baselines3.common.logger import configure

from soku_rl.env import EpisodeConfig,TwoPlayerVectorEnv
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.wrappers.learning import LearningConfig,LearningVectorEnv
from soku_rl.env.encoding import encode_action
from soku_rl.rl.matchup_env import MatchupMixtureVecEnv
from soku_rl.marl.br import OpponentEntry
from soku_rl.policy.loader import load_policy
from soku_rl.policy.contract import read_training_contract
from soku_rl.policy.god.package import ScriptPackage
from soku_rl.policy.god.continuous import ContinuousSourceTeacher
from soku_rl.policy.god.native_feedback import NativeTeacherFeedback
from soku_rl.env.observation.shared_native import SCHEMA,decode_shared
from soku_rl.rl.continuous_expert import validated_native_source
from soku_rl.rl.ppo import parameter_hash
from soku_rl.rl.training import EpisodeRecords
from soku_rl.rl.curriculum import FixedOpponentSchedule
from soku_rl.rl.storage import PackedObservation
from soku_rl.rl.takeover_training import (TakeoverSchedule,Demonstrations,ActorBC,
    load_student,save_bundle,restore_bundle,digest)

ROOT=Path(__file__).resolve().parents[2]


@contextmanager
def game_view(cfg, out, role, seed):
    with ExitStack() as stack:
        episode=EpisodeConfig.from_dict(cfg['episode'])
        backend=WorkerBackend(log_path=out/f'{role}-worker.log',**cfg['runtime'])
        stack.callback(backend.close);backend.configure_observation(episode.backend_observation())
        env=LearningVectorEnv(TwoPlayerVectorEnv(backend,1,episode),LearningConfig(**cfg['wrappers']))
        stack.callback(env.close)
        entry=cfg['algorithm']['opponents'][0]
        opponent=OpponentEntry(entry['name'],load_policy(entry['name'],entry['policy'],env.interface,'cpu'))
        view=MatchupMixtureVecEnv(env,0,[opponent],[1.],seed,
            cfg['algorithm']['matchups']['learner'],[entry['setup']])
        stack.callback(view.close);view.seed(seed)
        yield view,env.interface,backend


def auxiliary_episode(model,cfg,spec,out,identity,seed,teacher,buffer,frame_cap,proposal_interval=1):
    """One whole episode; never adds to or trains the PPO rollout buffer."""
    role='teacher' if teacher else 'evaluation'
    counts=dict(student=0,teacher=0,student_fallback=0)
    result=dict(role=role,status='RUNNING',seed=seed,actor_source_counts=counts)
    original=parameter_hash(model.policy);ppo_steps=model.num_timesteps;updates=model._n_updates
    rollout_position=(model.rollout_buffer.pos,model.rollout_buffer.full)
    mask=tuple(i not in model.policy.masked_actions for i in range(90))
    stream=(out/f'{role}-{seed}.pkl').open('xb')
    try:
        with game_view(cfg,out,role,seed) as (view,interface,backend):
            view.auto_reset=False
            obs=view.reset()
            if teacher:
                package=ScriptPackage(spec['package'],spec['api_source'])
                actor=ContinuousSourceTeacher(package,spec['script'],seed,interface,
                    initial_action=interface.action(encode_action((0,)*8)),instruction_limit=spec['instruction_limit'])
                native=decode_shared(obs[0],schema=SCHEMA)
                join=NativeTeacherFeedback(interface,0,view.reset_infos[0]['segment'],
                    initial_facing=int(native.players[0]['dir']))
                receipt=None;boundary=False
            for step in range(min(cfg['episode']['max_frames'],frame_cap)):
                old=obs[0].copy()
                advice=(actor.query(old,schema=SCHEMA,receipt=receipt,
                    trusted_boundary=boundary,legal_actions=mask) if teacher else None)
                label=advice['training_label'] if teacher else None
                valid=type(label) is int and 0<=label<90 and mask[label]
                # Only infer during teacher control when needed for fallback
                # or an explicitly requested diagnostic sample.
                diagnostic=proposal_interval>0 and step%proposal_interval==0
                proposal=(int(model.predict(obs,deterministic=False)[0][0])
                    if not teacher or not valid or diagnostic else None)
                if proposal is not None and not mask[proposal]:
                    raise RuntimeError('student selected a masked action')
                source=('teacher' if valid else 'student_fallback') if teacher else 'student'
                actual=label if valid else proposal
                obs,rewards,dones,infos=view.step(np.asarray([actual]))
                info=infos[0]
                consumed=False;receipt_data=None;evidence=None
                if teacher:
                    following=info['terminal_observation'] if dones[0] else obs[0]
                    native=decode_shared(following,schema=SCHEMA)
                    receipt,boundary,evidence=join.observe(step+1,info['native_combat_events'],actual,actor,native)
                    consumed=bool(receipt.consumed)
                    receipt_data=asdict(receipt)
                    if dones[0]:
                        actor.query(following,schema=SCHEMA,receipt=receipt,terminal=True)
                counts[source]+=1
                row=dict(schema=1,frame=step,episode=info['episode_id'],source=source,
                    student_proposal=proposal,teacher_suggestion=label,executed_action=actual,
                    student_proposal_recorded=proposal is not None,
                    valid=bool(valid and consumed),receipt=receipt_data,evidence=evidence,
                    advice=advice,policy_hash=original)
                # Store actual temporal context even when a teacher label is masked.
                if teacher:
                    row['obs']=PackedObservation.pack(old)
                    context=dict(frame=step,seed=seed,episode=info['episode_id'],trace=str(stream.name),offset=stream.tell())
                    buffer.add(old,actual,actor_source=source,valid=row['valid'],context=context)
                pickle.dump(row,stream,protocol=5)
                if dones[0]:
                    outcome=('timeout' if info['source_truncated'] else 'win' if info['outcome']=='p1_win'
                        else 'draw' if info['outcome']=='double_ko' else 'loss')
                    result.update(status='PASS_EPISODE',frames=step+1,outcome=outcome,
                        terminal_outcome=info['outcome'],reward_summary=info.get('combat_reward_summary'),
                        teacher_frames=counts['teacher'],policy_hash=original,
                        native_teacher=actor.report() if teacher else None)
                    break
            else:
                raise RuntimeError('auxiliary phase hit explicit frame cap without a terminal')
        assert parameter_hash(model.policy)==original and model._n_updates==updates and model.num_timesteps==ppo_steps
        assert (model.rollout_buffer.pos,model.rollout_buffer.full)==rollout_position
        result['ppo_buffer_frames_added']=0
        return result
    except BaseException as error:
        result.update(status='FAIL',error=repr(error));raise
    finally:
        stream.close()
        (out/f'{role}-{seed}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'rl/config/train_takeover.yaml'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume')
    args=parser.parse_args()
    out=Path(args.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('output must stay inside project')
    out.mkdir(parents=True,exist_ok=False)
    settings=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    cfg=OmegaConf.to_container(OmegaConf.load(settings['training_config']),resolve=True)
    options=settings['takeover']
    reset_method=options.get('cpu_reset_method','process_restart')
    if reset_method not in ('fresh','process_restart'):
        raise ValueError('explicit CPU reset method must be fresh or process_restart')
    cfg['runtime']['fresh_reset']=reset_method=='fresh'
    proposal_interval=options.get('student_proposal_interval',1)
    if type(proposal_interval) is not int or proposal_interval<0:
        raise ValueError('student_proposal_interval must be a nonnegative integer')
    if (type(options['enabled']) is not bool or options['schedule']!='independent_recent_win_streak_v2'
            or options['segment_boundary']!='episode'
            or any(type(options[k]) is not int or options[k]<1 for k in
                ('cycles','student_rollout_quota','evaluation_interval','teacher_bc_updates_per_cycle','bc_batch_size'))):
        raise ValueError('explicit bounded dual-stream configuration required')
    if cfg['episode']['decision_frames']!=1 or cfg['episode']['latency_frames']!=0:
        raise ValueError('only existing1F/no-latency contract supported')
    spec=cfg['rl']['kickstarting']['teacher']
    if spec['script']!=options['teacher_identity']:raise ValueError('teacher script mismatch')
    identity=dict(schema=1,config=settings,teacher_validation=validated_native_source(spec),
        package=ScriptPackage(spec['package'],spec['api_source']).original_fingerprint,
        source_checkpoint=digest(settings['student_checkpoint']),source_config=digest(settings['training_config']),
        input_contract='unchanged source checkpoint/90actions; no Human-BC history migration',
        cpu_reset_policy=reset_method,
        game_allocation='one student PPO instance plus one sequential auxiliary teacher/evaluation instance',
        algorithm='upstream PPO on student only;actor-only BC after consumed PPO rollout')
    identity['implementation']={name:digest(Path(__file__).parents[1]/name) for name in
        ('src/soku_rl/rl/takeover_training.py','src/soku_rl/rl/online_teacher.py','tools/train_takeover.py',
         'tools/native_cpu_worker.py','tools/game_runtime/observation.py')}
    (out/'config.json').write_text(json.dumps(dict(settings=settings,source_training=cfg,identity=identity),indent=2),encoding='utf-8')
    os.environ['SOKURL_CPU_RESET_PROCESS_RESTART']='0' if reset_method=='fresh' else '1'
    os.environ['SOKURL_NATIVE_CPU_FRESH_RESET']='1' if reset_method=='fresh' else '0'
    torch.set_num_threads(cfg['rl'].get('cpu_threads',4))
    started=time.perf_counter();report=dict(status='RUNNING',cycles=[])
    try:
        with game_view(cfg,out,'student',options['seed']) as (view,interface,backend):
            read_training_contract(settings['training_config'],interface)
            if args.resume:
                manifest=Path(args.resume).resolve()
                if not manifest.is_relative_to(ROOT):raise ValueError('local checkpoint only')
                model,bc,state=restore_bundle(manifest,view,cfg['device'],identity,
                    options['teacher_bc_coef'],options['bc_learning_rate'])
            else:
                model=load_student(settings['student_checkpoint'],view,cfg['device'])
                bc=ActorBC(model,options['teacher_bc_coef'],options['bc_learning_rate'])
                state=dict(cycle=0,phase='ppo',evaluations=[],rng=np.random.default_rng(options['seed']).bit_generator.state,
                    demos=Demonstrations(options['demo_capacity'],options['demo_max_bytes']),
                    native_frames=0,phases=[],bundle_serial=0)
            report.update(initial_policy_hash=parameter_hash(model.policy),source_steps=model.num_timesteps,
                ppo_class=type(model).__name__,observation_shape=model.observation_space.shape)
            model.set_logger(configure(str(out/'scalars'),['csv']))
            records=EpisodeRecords(out,32768,FixedOpponentSchedule(),cfg['episode']['combat_reward']['scale'])
            rng=np.random.default_rng();rng.bit_generator.state=state['rng']
            def checkpoint():
                state['rng']=rng.bit_generator.state
                state['bundle_serial']+=1
                path=save_bundle(out/f"checkpoint-{state['bundle_serial']:04d}",model,bc,state,identity)
                (out/'latest.json').write_text(json.dumps(dict(manifest=str(path),phase=state['phase'],cycle=state['cycle'])),encoding='utf-8')
            checkpoint()
            while state['cycle']<options['cycles']:
                if state['phase']=='ppo':
                    before=parameter_hash(model.policy);steps=model.num_timesteps
                    previous_updates=model._n_updates
                    model.learn(model.n_steps*options['student_rollout_quota'],callback=records,reset_num_timesteps=False)
                    advanced=model.num_timesteps-steps;state['native_frames']+=advanced
                    if advanced!=model.n_steps*options['student_rollout_quota']:raise RuntimeError('student rollout quota not fulfilled')
                    if model._n_updates<=previous_updates or parameter_hash(model.policy)==before:
                        raise RuntimeError('student-only PPO did not update parameters')
                    state['phases'].append(dict(phase='ppo',frames=advanced,actor_source='student',
                        before=before,after=parameter_hash(model.policy),ppo_epochs=model._n_updates))
                    model.rollout_buffer.reset()
                    state['phase']='teacher';checkpoint()
                if state['phase']=='teacher':
                    alpha=TakeoverSchedule(state['evaluations']).snapshot()
                    selected=bool(options['enabled'] and rng.random()<alpha['alpha'])
                    state['phases'].append(dict(phase='takeover_selection',schedule=alpha,selected=selected))
                    if selected:
                        episode=auxiliary_episode(model,cfg,spec,out,identity,int(rng.integers(1,2**31)),True,state['demos'],
                            options['native_frame_cap']-state['native_frames'],proposal_interval)
                        state['native_frames']+=episode['frames'];state['phases'].append(episode)
                        if not state['demos'].rows:
                            raise RuntimeError('BLOCKED: no trustworthy teacher-executed BC samples')
                    state['phase']='bc';checkpoint()
                if state['phase']=='bc':
                    while state.get('bc_cycle_updates',0)<options['teacher_bc_updates_per_cycle'] and state['demos'].rows:
                        x,y,ids=state['demos'].sample(rng,options['bc_batch_size'],model.device)
                        metric=bc.step(x,y);metric.update(phase='bc',sample_ids=ids)
                        state['phases'].append(metric)
                        state['bc_cycle_updates']=state.get('bc_cycle_updates',0)+1
                        checkpoint()
                    state['bc_cycle_updates']=0
                    model.rollout_buffer.reset()
                    state['phase']='evaluation';checkpoint()
                if state['phase']=='evaluation':
                    if (state['cycle']+1)%options['evaluation_interval']==0:
                        episode=auxiliary_episode(model,cfg,spec,out,identity,int(rng.integers(1,2**31)),False,state['demos'],
                            options['native_frame_cap']-state['native_frames'])
                        state['native_frames']+=episode['frames'];state['phases'].append(episode)
                        schedule=TakeoverSchedule(state['evaluations'])
                        schedule.add(f"evaluation-{len(state['evaluations'])}",episode['outcome'],
                            teacher_frames=episode['teacher_frames'],policy_hash=episode['policy_hash'])
                        state['evaluations']=schedule.history
                    state['cycle']+=1;state['phase']='ppo';checkpoint()
                if state['native_frames']>options['native_frame_cap']:raise RuntimeError('native frame cap exceeded')
            report.update(status='PASS_SHORT_DUALSTREAM',final_policy_hash=parameter_hash(model.policy),
                native_frames=state['native_frames'],cycle=state['cycle'],phases=state['phases'],
                schedule=TakeoverSchedule(state['evaluations']).snapshot(),bc_updates=bc.updates,
                final_ppo_steps=model.num_timesteps,final_ppo_epochs=model._n_updates,
                demo_rows=len(state['demos'].rows),demo_bytes=state['demos'].bytes)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        report['seconds']=time.perf_counter()-started
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

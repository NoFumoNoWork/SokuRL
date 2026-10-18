"""Continuous teacher attachment over existing vector observations and receipts.

The post-step packet is staged,not consulted for the current teacher label.
Only after that label is fixed is the real executed step reconciled for the
next decision. Environment actions and rewards are never replaced.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from soku_rl.env.encoding import decode_action,encode_action
from soku_rl.env.observation.shared_native import SCHEMA,SHARED_FEATURES,decode_shared
from soku_rl.policy.god.continuous import ContinuousSourceTeacher
from soku_rl.policy.god.native_feedback import NativeTeacherFeedback
from soku_rl.policy.god.package import ScriptPackage


PROVIDER='continuous_shared_v1'


def validate_spec(spec):
    required={'provider','package','api_source','script','seed','instruction_limit','validation'}
    if not isinstance(spec,dict) or set(spec)!=required or spec['provider']!=PROVIDER:
        raise ValueError('continuous_shared_v1 requires explicit pinned source and native validation')
    if spec['script'] not in ('01_marisa_main.ai','00_reimu_main.ai'):
        raise ValueError('reviewed original source required')
    if type(spec['seed']) is not int or not 0<=spec['seed']<2**32:
        raise ValueError('teacher seed must be uint32')
    if type(spec['instruction_limit']) is not int or not 100<=spec['instruction_limit']<=1000000:
        raise ValueError('bounded teacher instruction limit required')
    if any(not isinstance(spec[k],str) or not spec[k] for k in ('package','api_source','validation')):
        raise ValueError('explicit source/validation paths required')


def validated_native_source(spec):
    """Bind the tested interpreter/join,not a claim of all-branch game coverage."""
    root=Path(__file__).resolve().parents[4]
    path=Path(spec['validation']).resolve(strict=True)
    if not path.is_relative_to(root):raise ValueError('project-local native evidence required')
    report=json.loads(path.read_text('utf-8-sig'))
    if (report['status']!='PASS_EPISODE_NATIVE_SIDECAR' or
            report['profile']['script']!=spec['script'] or not report['student_only'] or
            report['teacher_game_writes'] or report['training_steps'] or
            report['initial_parameter_hash']!=report['final_parameter_hash']):
        raise ValueError('completed student-only native episode validation required')
    read=lambda p:json.loads(p.read_text('utf-8-sig'))
    before=read(path.parent/'before.json')
    if before!=read(path.parent/'after.json') or not read(path.parent.parent/'session/deployment.json')['restored']:
        raise ValueError('native validation restoration failed')
    if report['episodes'][0]['pids']!=report['episodes'][1]['pids']:
        raise ValueError('same-process native Reset validation required')
    if set(report['episodes'][0]['endings'])!={'0','1'}:
        raise ValueError('both-seat terminal validation required')
    keys=('src/soku_rl/policy/god/continuous.py','src/soku_rl/policy/god/native_feedback.py',
          'src/soku_rl/policy/god/shared_native.py','src/soku_rl/env/observation/shared_native.py')
    for key in keys:
        actual=hashlib.sha256((root/'rl'/key).read_bytes()).hexdigest()
        if before['source_hashes'].get('RL/'+key)!=actual:
            raise ValueError('stale native teacher validation: '+key)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContinuousExpert:
    def __init__(self,interface,spec,num_envs):
        validate_spec(spec)
        if (not getattr(interface.episode,'shared_teacher_context',False) or
                interface.observation_space.shape!=(SHARED_FEATURES,) or
                len(interface.commands)!=90 or
                (interface.episode.decision_frames,interface.episode.latency_frames)!=(1,0)):
            raise ValueError('continuous expert requires the explicit shared1F/90 contract')
        self.spec=dict(spec)
        validation=validated_native_source(spec)
        self.package=ScriptPackage(spec['package'],spec['api_source'])
        self.fingerprint=hashlib.sha256(json.dumps(dict(spec=spec,validation=validation,
            package=self.package.original_fingerprint,
            implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),sort_keys=True).encode()).hexdigest()
        self.interface,self.num_envs=interface,num_envs
        self.mask=tuple(not any(decode_action(interface.command(i)).inputs[-2:]) for i in range(90))
        self.neutral=interface.action(encode_action((0,)*8))
        self.pending=None
        self.actors=[None]*num_envs
        self.joins=[None]*num_envs
        self.receipts=[None]*num_envs
        self.boundaries=[False]*num_envs
        self.counts=Counter()
        self.closed=[]

    def reset(self):
        # Called only AFTER the underlying environment really reset.
        self.pending=None
        self.actors=[None]*self.num_envs
        self.joins=[None]*self.num_envs
        self.receipts=[None]*self.num_envs
        self.boundaries=[False]*self.num_envs

    def stage(self,observations,infos,dones):
        if self.pending is not None:
            raise RuntimeError('unconsumed native step;reset before restarting collection')
        if len(infos)!=self.num_envs or np.shape(dones)!=(self.num_envs,):
            raise ValueError('one native feedback packet per student slot required')
        self.pending=(observations.copy(),[dict(i) for i in infos],np.asarray(dones).copy())

    def labels(self,observations,starts,actions,state):
        if self.pending is None:raise RuntimeError('native post-step feedback was not staged')
        following,infos,dones=self.pending
        labels=np.zeros(self.num_envs,np.int64)
        valid=np.zeros(self.num_envs,bool)
        actions=np.asarray(actions).reshape(self.num_envs)
        # Label every OLD state before inspecting any following state/events.
        for slot in range(self.num_envs):
            if starts[slot]:
                if self.actors[slot] is not None and not self.actors[slot].ended:
                    raise RuntimeError('episode start without closing the preceding teacher')
                old=decode_shared(observations[slot],schema=SCHEMA)
                if old.world['frame']!=0:raise ValueError('teacher must start at real frame0')
                seat=infos[slot]['learner_seat']
                if type(seat) is not int or seat not in (0,1):raise ValueError('actual learner seat required')
                seed=(self.spec['seed']+state['episodes'])%2**32
                self.actors[slot]=ContinuousSourceTeacher(self.package,self.spec['script'],seed,
                    self.interface,initial_action=self.neutral,instruction_limit=self.spec['instruction_limit'])
                self.joins[slot]=NativeTeacherFeedback(self.interface,seat,infos[slot]['segment'],
                    initial_facing=int(old.players[0]['dir']))
                self.receipts[slot]=None;self.boundaries[slot]=False
                state['episodes']+=1
            actor=self.actors[slot]
            if actor is None:raise RuntimeError('missing teacher episode prefix')
            advice=actor.query(observations[slot],schema=SCHEMA,receipt=self.receipts[slot],
                trusted_boundary=self.boundaries[slot],legal_actions=self.mask)
            label=advice['training_label']
            valid[slot]=label is not None
            if label is not None:
                labels[slot]=label;state['valid']+=1
                state['matches']+=int(label==actions[slot])
            status=advice['reason']
            state['statuses'][status]=state['statuses'].get(status,0)+1
            if len(state['examples'])<32:state['examples'].append(advice)
        # No advice for this decision can depend on the completed step below.
        for slot in range(self.num_envs):
            actor=self.actors[slot];info=infos[slot]
            after=info['terminal_observation'] if dones[slot] else following[slot]
            native=decode_shared(after,schema=SCHEMA)
            receipt,boundary,_=self.joins[slot].observe(int(native.world['frame']),
                info['native_combat_events'],int(actions[slot]),actor,native)
            self.receipts[slot],self.boundaries[slot]=receipt,boundary
            self.counts['native_steps']+=1
            self.counts['verified_consumed_steps']+=int(receipt.consumed)
            self.counts['matched_acceptance_receipts']+=int(receipt.accepted)
            if dones[slot]:
                final=actor.query(after,schema=SCHEMA,receipt=receipt,terminal=True)
                if final['reason']!='EPISODE_TERMINAL' or final['training_label'] is not None:
                    raise RuntimeError('untrustworthy terminal teacher closure')
                self.closed.append(actor.report());self.closed=self.closed[-16:]
                self.counts['terminal_closures']+=1
        self.pending=None
        state['queries']+=self.num_envs;state['elapsed']+=self.num_envs
        state['native_guidance']=dict(counts=dict(self.counts),
            current=[None if a is None else a.report() for a in self.actors],closed=self.closed)
        return labels,valid


class FeedbackVecEnv(VecEnvWrapper):
    """Transport existing post-step evidence without controlling inputs/rewards."""
    def __init__(self,env,expert):
        super().__init__(env)
        self.expert=expert

    def reset(self):
        values=self.venv.reset()
        self.reset_infos=self.venv.reset_infos
        self.expert.reset()
        return values

    def step_wait(self):
        result=self.venv.step_wait()
        self.expert.stage(result[0],result[3],result[2])
        return result

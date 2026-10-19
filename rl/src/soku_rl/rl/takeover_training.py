"""Two disjoint data streams, one student policy; no teacher PPO transitions."""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3 import PPO

from soku_rl.rl.buffers import PackedRolloutBuffer
from soku_rl.rl.storage import PackedObservation
from soku_rl.rl.online_teacher import actor_parameters


class TakeoverSchedule:
    """Latest user formula, without the old sample-count factor or .95 cap."""
    def __init__(self, history=()):
        self.history = list(history)
        identifiers = [row['id'] for row in self.history]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('duplicate independent evaluation')
        if any(type(row['won']) is not bool or row['teacher_frames'] != 0 for row in self.history):
            raise ValueError('only teacher-free completed evaluations count')

    def add(self, identifier, outcome, *, teacher_frames, policy_hash):
        if teacher_frames != 0 or outcome not in ('win','loss','draw','timeout'):
            raise ValueError('independent evaluation required')
        if any(row['id']==identifier for row in self.history):
            raise ValueError('evaluation already counted')
        self.history.append(dict(id=identifier,won=outcome=='win',outcome=outcome,
            teacher_frames=0,policy_hash=policy_hash))

    def snapshot(self):
        recent = self.history[-10:]
        streak = longest = 0
        for row in recent:
            streak = streak+1 if row['won'] else 0
            longest = max(longest,streak)
        n = len(recent)
        wins = sum(row['won'] for row in recent)
        alpha = 1. if not n else float(np.clip(1.-wins/(2*n)-.1*min(longest,5),0.,1.))
        return dict(t=len(self.history),n=n,wins=wins,longest_streak=longest,alpha=alpha)


class Demonstrations:
    def __init__(self, capacity, max_bytes):
        if type(capacity) is not int or capacity<1 or type(max_bytes) is not int or max_bytes<1:
            raise ValueError('positive demonstration bounds required')
        self.capacity,self.max_bytes=capacity,max_bytes
        self.rows=deque();self.bytes=0;self.next_id=0;self.evicted=0

    def add(self, observation, action, *, actor_source, valid, context):
        if actor_source!='teacher' or not valid:
            return False
        if type(action) is not int or not 0<=action<90:
            raise ValueError('representable teacher action required')
        packed=PackedObservation.pack(observation)
        if len(packed.content)>self.max_bytes:
            raise RuntimeError('one demonstration exceeds explicit memory bound')
        row=dict(id=self.next_id,obs=packed,action=action,context=dict(context))
        self.next_id+=1;self.rows.append(row);self.bytes+=len(packed.content)
        while len(self.rows)>self.capacity or self.bytes>self.max_bytes:
            self.bytes-=len(self.rows.popleft()['obs'].content);self.evicted+=1
        return True

    def sample(self, rng, count, device):
        if not self.rows:
            raise RuntimeError('no valid teacher-executed demonstrations')
        indices=rng.choice(len(self.rows),size=min(count,len(self.rows)),replace=False)
        rows=[self.rows[int(i)] for i in indices]
        observations=torch.as_tensor(np.stack([r['obs'].unpack() for r in rows]),device=device)
        actions=torch.as_tensor([r['action'] for r in rows],device=device,dtype=torch.long)
        return observations,actions,[r['id'] for r in rows]


class ActorBC:
    def __init__(self, model, coefficient, learning_rate):
        if (not math.isfinite(coefficient) or coefficient<0 or
                not math.isfinite(learning_rate) or learning_rate<=0):
            raise ValueError('independent finite BC coefficient and learning rate required')
        self.model,self.coefficient=model,coefficient
        self.parameters=actor_parameters(model.policy)
        # Same momentum-free SGD mechanism as online_teacher, actor-only set.
        self.optimizer=torch.optim.SGD(self.parameters,lr=learning_rate,momentum=0.,weight_decay=0.)
        self.updates=0

    def step(self, observations, actions):
        if self.coefficient==0:
            return dict(updated=False,nll=None)
        self.model.policy.set_training_mode(True)
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        critic=[*self.model.policy.value_net.parameters(),*self.model.policy.mlp_extractor.value_net.parameters()]
        if not self.model.policy.share_features_extractor:
            critic.extend(self.model.policy.vf_features_extractor.parameters())
        before=[p.detach().clone() for p in critic]
        log_prob=self.model.policy.get_distribution(observations).log_prob(actions)
        nll=-log_prob.mean()
        if not torch.isfinite(nll):
            raise RuntimeError('non-finite teacher-executed BC loss')
        (self.coefficient*nll).backward()
        if any(p.grad is not None for p in critic):
            raise RuntimeError('BC reached Critic-exclusive parameters')
        norm=torch.nn.utils.clip_grad_norm_(self.parameters,self.model.max_grad_norm,error_if_nonfinite=True)
        self.optimizer.step();self.updates+=1
        if any(not torch.equal(old,p) for old,p in zip(before,critic,strict=True)):
            raise RuntimeError('BC modified Critic-exclusive parameters')
        self.optimizer.zero_grad(set_to_none=True)
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        return dict(updated=True,nll=float(nll.detach()),gradient_norm=float(norm),updates=self.updates)


def load_student(path, env, device):
    """Explicit algorithm migration; preserve policy/Adam, discard old expert loss.

    This is not a network migration. Loading via PPO selects upstream train(),
    and ordinary packed storage prevents old expert queries during rollout.add.
    """
    model=PPO.load(path,env=env,device=device,rollout_buffer_class=PackedRolloutBuffer,
                   rollout_buffer_kwargs={})
    if type(model) is not PPO or model.action_space.n!=90:
        raise ValueError('ordinary categorical90 PPO required')
    for key in ('kickstarting_state','teacher_state'):
        if hasattr(model,key):delattr(model,key)
    if hasattr(model,'_expert') or hasattr(model,'_online_teacher'):
        raise ValueError('old expert attachment must not survive migration')
    return model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_bundle(directory, model, bc, state, identity):
    """Write new, complete phase-boundary bundles; never overwrite evidence."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    model.save(directory/'student.zip')
    payload=dict(state=state,bc_optimizer=bc.optimizer.state_dict(),bc_updates=bc.updates,
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.save(payload,directory/'state.pt')
    manifest=dict(schema=1,identity=identity,
        files={p.name:digest(p) for p in (directory/'student.zip',directory/'state.pt')},
        resume='completed-phase boundary; fresh native student episode, no live Lua plan or PPO rollout restored')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return directory/'manifest.json'


def restore_bundle(manifest_path, env, device, identity, coefficient, learning_rate):
    path=Path(manifest_path);manifest=json.loads(path.read_text())
    if manifest['schema']!=1 or manifest['identity']!=identity:
        raise ValueError('checkpoint identity/contract mismatch')
    for name,expected in manifest['files'].items():
        if Path(name).name!=name or digest(path.parent/name)!=expected:
            raise ValueError('checkpoint hash mismatch')
    model=load_student(path.parent/'student.zip',env,device)
    bc=ActorBC(model,coefficient,learning_rate)
    # Trusted project-local bundle only, not an arbitrary downloaded pickle.
    data=torch.load(path.parent/'state.pt',map_location='cpu',weights_only=False)
    bc.optimizer.load_state_dict(data['bc_optimizer']);bc.updates=data['bc_updates']
    random.setstate(data['python_rng']);np.random.set_state(data['numpy_rng'])
    torch.set_rng_state(data['torch_rng'])
    if data['cuda_rng'] is not None:
        if not torch.cuda.is_available():raise ValueError('saved CUDA RNG requires CUDA')
        torch.cuda.set_rng_state_all(data['cuda_rng'])
    return model,bc,data['state']

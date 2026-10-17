"""Causal public event history and conservative human command-onset labels."""
from collections import deque
import re

import numpy as np

from sokurl_platform.replay_inputs import decode_keys
from sokurl_platform.command_input import DIRECTIONS
from soku_rl.policy.hierarchical_control import command_catalog


LATEST=slice(1295,1480)


class PublicEventHistory:
    """Identical offline/live feature builder; no native pointers/opponent keys."""
    def __init__(self):self.reset()

    def reset(self):
        self.events=deque(maxlen=16);self.inputs=deque(maxlen=60)
        self.previous=None;self.frame=None;self.previous_input=None;self.held=0
        self.action_since=[None,None]

    def observe(self,state,frame,last_own_input=None):
        state=np.asarray(state,np.float32)
        if state.shape!=(185,) or not np.isfinite(state).all():raise ValueError('public185state required')
        if self.frame is not None and frame!=self.frame+1:raise ValueError('reset public history on gap/round')
        changed=False
        if last_own_input is not None:
            keys=np.asarray(last_own_input,np.float32)
            if keys.shape!=(8,):raise ValueError('only own8logicalchannels')
            self.inputs.append(keys.copy());self.held+=1
            if self.previous_input is not None and not np.array_equal(keys,self.previous_input):
                self.events.append((3,0,0,frame,self.held-1,0.,state.copy()));self.held=1;changed=True
            self.previous_input=keys.copy()
        if self.previous is not None:
            for owner in (0,1):
                offset=owner*10
                # Missing visibility does not create a fabricated native event.
                if state[offset] and self.previous[offset]:
                    old,new=int(self.previous[offset+5]),int(state[offset+5])
                    if old!=new:
                        if self.action_since[owner] is not None:
                            self.events.append((4,old,owner,frame,frame-self.action_since[owner],0.,state.copy()))
                        self.events.append((1,new,owner,frame,0,0.,state.copy()));changed=True
                        self.action_since[owner]=frame
                    elif self.action_since[owner] is None:self.action_since[owner]=frame-1
                else:
                    self.action_since[owner]=frame if state[offset] else None
                delta=float(state[-5+owner]-self.previous[-5+owner])
                if delta:
                    self.events.append((2,0,owner,frame,0,delta,state.copy()));changed=True
        self.previous=state.copy();self.frame=frame
        return changed

    def tensors(self):
        ids=np.zeros((16,3),np.int64);states=np.zeros((16,185),np.float32)
        numbers=np.zeros((16,3),np.float32);mask=np.zeros(16,bool)
        for i,event in enumerate(self.events,16-len(self.events)):
            kind,action,owner,frame,duration,result,state=event
            ids[i]=kind,action,owner;states[i]=state;mask[i]=True
            numbers[i]=(np.log1p(self.frame-frame)/8,np.log1p(duration)/8,result)
        inputs=np.zeros((60,8),np.float32)
        if self.inputs:inputs[-len(self.inputs):]=np.stack(self.inputs)
        return dict(state=self.previous.copy(),event_ids=ids,event_states=states,
            event_numbers=numbers,event_mask=mask,inputs=inputs,
            history_known=np.array(len(self.inputs)==60))


def common_bindings(path):
    """Parse community motion names, not a character-specific runtime handler.

    Skills/charged variants without an unambiguous command name stay unsupported.
    AA stages label a SINGLE extra A press, never an entire AAA macro.
    """
    bindings={}
    text=path.read_text(encoding='utf-8-sig')
    for line in text.splitlines():
        match=re.match(r'^\s*(\S+)\s+(\d+)\s*$',line)
        if not match:continue
        name,native=match.group(1),int(match.group(2))
        if re.fullmatch(r'A{2,5}',name):bindings[native]='5A';continue
        name=name.removeprefix('J')
        if re.fullmatch(r'[1-9]?[ABC]',name):
            bindings[native]=name if len(name)>1 else '5'+name
    return bindings


def onset_label(arrays,decision,bindings,catalog=None,max_lookback=30):
    """Unique consumed raw input edge joined to accepted action-name evidence.

    This is not proof of which EXE buffered key was consumed or latent intention.
    Ambiguous/cross-gap/charge/unsupported skill cases are masked,never replaced.
    """
    catalog=catalog or command_catalog()
    name=bindings.get(decision['action_id']);names=[c['name'] for c in catalog]
    if name not in names:return dict(status='UNKNOWN_COMMAND',onset=None,label=-1)
    row=decision['observation_row'];seat=int(arrays['seats'][row])
    desired=DIRECTIONS[int(name[0])];button={'A':2,'B':3,'C':4}[name[-1]]
    candidates=[]
    for j in range(max(1,row-max_lookback),row+1):
        if not (arrays['raw_words_present'][j-1:j+1].all() and arrays['compatible'][j:row+1].all()
                and not arrays['boundary'][j-1:row+1].any()
                and arrays['raw_words_present'][j:row+1].all()):continue
        before=tuple(decode_keys(int(arrays['raw_words'][j-1,seat])).values())
        keys=tuple(decode_keys(int(arrays['raw_words'][j,seat])).values())
        facing=float(arrays['observations'][j,LATEST][3])
        if facing not in (-1,1):continue
        relative=(int(keys[0]*facing),keys[1])
        if not before[button] and keys[button] and relative==desired and sum(keys[2:5])==1:
            candidates.append(j)
    if len(candidates)!=1:return dict(status='AMBIGUOUS_INPUT_EDGE',onset=None,label=-1)
    onset=candidates[0]
    # A distinct accepted operation between onset and acceptance disqualifies it
    # in the caller. Never turn a native action ID directly into logical keys.
    return dict(status='UNIQUE_RAW_EDGE_WITH_COMMUNITY_ACTION_AND_NATIVE_ACCEPTANCE',
        onset=onset,label=names.index(name),native_acceptance_row=row,
        limitation='Not instruction-level input-consumption proof')


def build_hierarchical_case(arrays,chains,bindings):
    catalog=command_catalog();names=[c['name'] for c in catalog]
    valid=(arrays['compatible']&~arrays['boundary']&arrays['raw_words_present'])
    choices=[];unknown=[]
    for decision in chains['decisions']:
        label=onset_label(arrays,decision,bindings,catalog)
        if label['onset'] is None:unknown.append(dict(decision_id=decision['id'],**label));continue
        j=label['onset'];end=decision['observation_row']
        if (decision['same_frame_acceptances']!=1 or any(j<=d['observation_row']<end
                for d in chains['decisions'] if d['id']!=decision['id'])):
            unknown.append(dict(decision_id=decision['id'],status='INTERVENING_OPERATION'));continue
        choices.append(dict(row=j,label=label['label'],evidence=label['status'],acceptance=end))
    for movement in chains['movements']:
        j=movement['start_row'];h,v,d=movement['logical'];facing=arrays['observations'][j,LATEST][3]
        if j<1 or facing not in (-1,1) or not valid[j-1:j+1].all():continue
        previous=tuple(decode_keys(int(arrays['raw_words'][j-1,int(arrays['seats'][j])])).values())
        current=(h,v,0,0,0,d,0,0)
        if previous==current:continue  # Not a fresh direction/release choice.
        direction=next(k for k,value in DIRECTIONS.items() if value==(int(h*facing),v))
        name=f'{direction}{"D" if d else "move"}'
        choices.append(dict(row=j,label=names.index(name),evidence='RAW_MOVEMENT_OR_RELEASE_EDGE',acceptance=None))
    # Same-onset ambiguous operations are not arbitrary class tie-breaks.
    counts={j:sum(c['row']==j for c in choices) for j in {c['row'] for c in choices}}
    choices=sorted((c for c in choices if counts[c['row']]==1),key=lambda c:c['row'])
    labelled_acceptances={c['acceptance'] for c in choices if c['acceptance'] is not None}
    unknown_rows={d['observation_row'] for d in chains['decisions']
                  if d['observation_row'] not in labelled_acceptances}
    unknown_rows.update(j for j,count in counts.items() if count>1)
    lookup={c['row']:i for i,c in enumerate(choices)}
    history=PublicEventHistory();rows=[];targets=[];audit=[];active=len(catalog);active_start=0
    for j in range(len(valid)):
        if not valid[j]:history.reset();active=len(catalog);continue
        if j in unknown_rows:active=len(catalog)
        previous=None
        if j and valid[j-1]:previous=tuple(decode_keys(int(arrays['raw_words'][j-1,int(arrays['seats'][j])])).values())
        changed=history.observe(arrays['observations'][j,LATEST],int(arrays['observation_frames'][j]),previous)
        selected=lookup.get(j)
        # Non-decision feedback can supervise continue,not an invented new action.
        if selected is None and not changed:continue
        features=history.tensors()
        if not features['history_known']:continue
        features.pop('history_known')
        features.update(candidates=np.ones(len(catalog),bool),active_action=np.array(active,np.int64),
            elapsed=np.array(j-active_start,np.float32))
        future=np.full(3,-1,np.int64);future_mask=np.zeros(3,bool)
        label=-1
        if selected is not None:
            label=choices[selected]['label']
            for offset,c in enumerate(choices[selected+1:selected+4]):
                if not valid[j:c['row']+1].all() or any(j<k<=c['row'] for k in unknown_rows):break
                future[offset]=c['label'];future_mask[offset]=True
        own_action=int(features['state'][5])
        forced=50<=own_action<200
        # No end-of-prefix stop label: that endpoint is censored.
        targets.append(dict(action=label,action_mask=selected is not None and not forced,
            termination=float(selected is not None),termination_mask=active<len(catalog) and not forced,
            sequence=future,sequence_mask=future_mask & (selected is not None and not forced)))
        rows.append(features);audit.append(dict(row=j,observation_frame=int(arrays['observation_frames'][j]),
            selection=choices[selected] if selected is not None else None,
            legality='CANDIDATE_VOCAB_ONLY_ENGINE_LEGAL_SET_NOT_RECORDED'))
        if selected is not None and not forced:active=label;active_start=j
        if forced:active=len(catalog)
    if not rows:return {},dict(choices=choices,unknown=unknown,samples=[])
    data={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    data.update({'target_'+k:np.stack([t[k] for t in targets]) for k in targets[0]})
    return data,dict(choices=choices,unknown=unknown,samples=audit)

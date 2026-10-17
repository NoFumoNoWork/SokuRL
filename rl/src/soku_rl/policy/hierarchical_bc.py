"""One episode's public event memory and native-interpreter command execution.

    The caller must provide existing control feedback and a verified legal set.
    This adapter never opens process memory or invents an engine legality oracle.
"""
import numpy as np
import torch

from soku_rl.replay.hierarchical import PublicEventHistory,LATEST
from soku_rl.policy.hierarchical_control import EventCommandController
from soku_rl.env.encoding import encode_action


class HierarchicalEpisode:
    def __init__(self,actor,catalog,interface,device='cpu',max_wait=120,legality_mode='strict'):
        if interface.episode.decision_frames!=1 or interface.episode.latency_frames!=0:
            raise ValueError('hierarchical executor requires existing1F zero-latency transport')
        if interface.action_space.n!=90:raise ValueError('preserve primitive90 action interface')
        self.actor=actor;self.device=device;self.interface=interface
        self.semantic_history=getattr(actor,'event_history_version',1)==2
        if self.semantic_history:
            from soku_rl.replay.semantic_events import SemanticEventHistory
            self.history=SemanticEventHistory()
        else:self.history=PublicEventHistory()
        self.controller=EventCommandController(catalog,max_wait,legality_mode)
        self.decisions=0;self.evaluations=0;self.last_input=None

    @torch.inference_mode()
    def act(self,observation,*,frame,facing,segment,events,seat,opportunity):
        value=np.asarray(observation,np.float32)
        if value.shape!=(1544,):raise ValueError('unchanged public1544 observation required')
        if not self.controller.accepts_source(opportunity.legality_source):
            raise RuntimeError('BLOCKED: existing Platform does not supply full high-level legal sets')
        if not hasattr(self,'segment'):self.segment=segment
        if self.segment!=segment:raise ValueError('new episode adapter required at reset')
        extra={}
        if self.semantic_history:
            from soku_rl.replay.semantic_events import contact_feedback
            extra['feedback']=contact_feedback(events,seat,frame)
        changed=self.history.observe(value[LATEST],frame,None if self.history.frame is None else value[-8:],**extra)
        actor_output=None
        if opportunity.can_respond and not opportunity.forced and (changed or
                opportunity.reason!='NONE' or self.controller.player is None or
                self.controller.completed or self.controller.elapsed>=self.controller.max_wait):
            data=self.history.tensors();data.pop('history_known')
            data.update(candidates=np.array(opportunity.legal_actions,bool),
                active_action=np.array(self.controller.active if self.controller.player else len(self.controller.catalog)),
                elapsed=np.array(self.controller.elapsed,np.float32))
            batch={k:torch.as_tensor(v,device=self.device).unsqueeze(0) for k,v in data.items()}
            self.actor.eval();actor_output=self.actor(batch);self.evaluations+=1
        stop=actor_output is not None and bool(actor_output['termination'][0].sigmoid()>=.5)
        def decide(_):
            if actor_output is None:raise RuntimeError('missing causal boundary inference')
            self.decisions+=1
            return (int(actor_output['choice_id'][0]) if 'choice_id' in actor_output
                    else int(actor_output['action'][0].argmax()))
        keys,replan=self.controller.tick(frame,facing,segment,events,seat,opportunity,
            stop=stop,decider=decide)
        primitive=self.interface.inverse.get(encode_action(keys))
        if primitive is None:raise RuntimeError('command cannot be represented by existing90 actions')
        return int(primitive),dict(replan=replan,accepted=self.controller.accepted,
            elapsed=self.controller.elapsed,active=self.controller.active,
            raw_logical_input=keys,legality_source=opportunity.legality_source,
            game_legality_known=opportunity.legality_source=='PROVIDED_ENGINE_LEGAL_SET')

"""Interruptible high-level commands over the unchanged Platform interpreter."""
from collections import deque
from dataclasses import dataclass

from sokurl_platform.command_input import CommandPlayer,Keys,compile_commands


def command_catalog():
    rows=[]
    for direction in range(1,10):
        for buttons in ('','D'):
            rows.append(dict(name=f'{direction}{buttons or "move"}',
                direction=direction,buttons=buttons,continuous=True))
    for direction in range(1,10):
        for button in 'ABC':
            rows.append(dict(name=f'{direction}{button}',command=f'{direction}{button}',continuous=False))
    for motion in ('236','623','214','22'):
        for button in 'BC':rows.append(dict(name=motion+button,command=motion+button,continuous=False))
    for command in ('2B{40}','5B{40}','5C{40}'):
        rows.append(dict(name=command,command=command,continuous=False))
    return rows


def program(entry):
    return ((Keys(entry['direction'],entry['buttons'],10000,1),) if entry['continuous']
            else compile_commands(entry['command']))


@dataclass(frozen=True)
class Opportunity:
    """Control-side mask; its source distinguishes legal proof from candidates."""
    can_respond: bool
    legal_actions: tuple[bool,...]
    reason: str
    legality_source: str
    forced: bool=False


class EventCommandController:
    def __init__(self,catalog=None,max_wait=120,legality_mode='strict'):
        self.catalog=catalog or command_catalog()
        if legality_mode not in ('strict','masked_native_feedback'):raise ValueError('explicit legality mode required')
        self.legality_mode=legality_mode
        if type(max_wait) is not int or not 1<=max_wait<=10000:raise ValueError('bounded watchdog required')
        self.max_wait=max_wait
        self.reset()

    def reset(self):
        self.player=None;self.active=None;self.elapsed=0;self.last_frame=None;self.segment=None
        self.feedback=[];self.completed=False;self.accepted=False;self.trace=deque(maxlen=64)
        self.attempts=[]

    def accepts_source(self,source):
        return source=='PROVIDED_ENGINE_LEGAL_SET' or (self.legality_mode=='masked_native_feedback'
            and source=='MASKED_CANDIDATES_NATIVE_AUTHORITY')

    def finish_attempt(self,reason):
        if self.attempts and self.attempts[-1]['closed_by'] is None:
            self.attempts[-1].update(closed_by=reason,
                acceptance='NATIVE_ACCEPTED' if self.accepted else 'NOT_OBSERVED',
                frames=self.elapsed)

    def observe_feedback(self,events,seat):
        if self.player:
            self.player.observe_native(events,seat)
            self.feedback=list(self.player.feedback)
            self.accepted=any(e['kind']=='ACCEPTED' for e in self.feedback)
            if self.attempts and self.accepted:self.attempts[-1]['acceptance']='NATIVE_ACCEPTED'

    def choose(self,index,opportunity):
        if (not opportunity.can_respond or opportunity.forced or
                not self.accepts_source(opportunity.legality_source)):
            raise RuntimeError('BLOCKED: authoritative legal response set required')
        if (type(index) is not int or not 0<=index<len(self.catalog) or
                len(opportunity.legal_actions)!=len(self.catalog) or not opportunity.legal_actions[index]):
            raise ValueError('illegal high-level choice')
        self.finish_attempt('REPLAN')
        self.attempts.append(dict(id=len(self.attempts),command=self.catalog[index]['name'],
            selected_frame=self.last_frame,mask_source=opportunity.legality_source,
            submitted=False,acceptance='NOT_OBSERVED',closed_by=None,frames=0))
        self.player=CommandPlayer(program(self.catalog[index]));self.active=index
        self.elapsed=0;self.accepted=False;self.completed=False

    def tick(self,frame,facing,segment,events,seat,opportunity,*,stop=False,decider=None):
        if self.last_frame is not None and (frame!=self.last_frame+1 or segment!=self.segment):
            raise RuntimeError('reset controller at episode/gap boundary')
        self.last_frame=frame;self.segment=segment
        if len(opportunity.legal_actions)!=len(self.catalog):raise ValueError('legal set shape mismatch')
        self.observe_feedback(events,seat)
        # Forced states advance time but do not create a policy choice.
        interrupted=opportunity.forced
        watchdog=self.elapsed>=self.max_wait
        replan=bool(opportunity.can_respond and not interrupted and
                    (self.player is None or self.completed or stop or watchdog))
        if interrupted:
            self.finish_attempt('FORCED_INTERRUPT')
            self.player=None;self.active=None;self.elapsed=0;self.completed=False
        elif replan:
            self.player=None;self.completed=False
            if decider is not None:self.choose(int(decider(opportunity)),opportunity)
        keys=(0,)*8 if self.player is None else self.player.next(frame,facing,segment=segment)
        if self.player:
            self.completed=self.player.done
            self.elapsed+=1
            self.attempts[-1].update(submitted=True,frames=self.elapsed)
        self.trace.append(dict(frame=frame,active=self.active,replan=replan,interrupted=interrupted,
            accepted=self.accepted,watchdog=watchdog,submitted=keys))
        return keys,replan


def candidate_opportunity(catalog,state,allowed):
    """Conservative execution policy using existing V3 control feedback.

    Not an engine legality classifier. The EXE still accepts or ignores inputs.
    Unknown states block new selection;cancel feedback uses the existing
    CommandPlayer/V3 gate rather than a new interpretation of native fields.
    """
    names={c['name'] for c in catalog}
    if not allowed or not set(allowed)<=names:raise ValueError('explicit known candidate subset required')
    action=state['action']
    forced=bool(state['hitstop'] or state['untech'] or 50<=action<180)
    ordinary=action in (0,1,2,3,4,5,6,7,8,9,200,201,202,203,204,208,209,210,211,212,214)
    cancel=bool(state['frame_flags']&0x20) and state['hitstop']==0
    responsive=not forced and (ordinary or cancel)
    mask=tuple(c['name'] in allowed and responsive for c in catalog)
    return Opportunity(responsive,mask,'IDLE_MOVEMENT_OR_EXISTING_CANCEL_GATE' if responsive else 'NONE',
        'MASKED_CANDIDATES_NATIVE_AUTHORITY',forced)

"""One frame-driven command player for every character; the EXE owns legality."""
from collections import deque
from dataclasses import dataclass,replace
import re

def encode_action(inputs):
    """Platform's existing 576-way logical-key encoding; no RL dependency."""
    if len(inputs) != 8 or any(type(v) is not int for v in inputs):
        raise ValueError('eight integer logical inputs required')
    if any(v not in (-1, 0, 1) for v in inputs[:2]) or any(v not in (0, 1) for v in inputs[2:]):
        raise ValueError('invalid logical inputs')
    return ((inputs[0] + 1) * 3 + inputs[1] + 1) * 64 + sum(
        value << bit for bit, value in enumerate(inputs[2:]))

NEUTRAL=(0,)*8
BUTTONS={'A':2,'B':3,'C':4,'D':5,'E':6,'F':7}
DIRECTIONS={1:(-1,1),2:(0,1),3:(1,1),4:(-1,0),5:(0,0),6:(1,0),
            7:(-1,-1),8:(0,-1),9:(1,-1)}


@dataclass(frozen=True)
class Keys:
    direction: int
    buttons: str
    frames: int
    command: int


@dataclass(frozen=True)
class WaitFor:
    condition: str
    timeout: int=120
    value: int | None=None
    retry: bool=False

    def __post_init__(self):
        if self.condition not in {'accepted','hit','block','idle','cancel_window','action'}:
            raise ValueError('unsupported command-chain gate')
        if type(self.timeout) is not int or not 1<=self.timeout<=10000:
            raise ValueError('positive bounded gate timeout required')
        if self.condition=='action' and (type(self.value) is not int or self.value<0):
            raise ValueError('action gate requires an explicit action ID')
        if type(self.retry) is not bool or (self.retry and self.condition!='accepted'):
            raise ValueError('retry is only supported for native acceptance gates')


def compile_commands(text, *, tap_frames=2, release_frames=2, charge_frames=40, confirm=False):
    """Digits are facing-relative; (N) neutral wait; 2HB or 2B{40} hold.

    AAA includes releases. Timing is an input policy, not a move frame table.
    This strict grammar intentionally does not execute arbitrary Lua.
    """
    if not isinstance(text,str) or len(text)>4096:
        raise ValueError('bounded command text required')
    if any(type(n) is not int or not 1<=n<=10000 for n in (tap_frames,release_frames,charge_frames)):
        raise ValueError('positive bounded input durations required')
    stages=[];pos=0;command=0
    token=re.compile(r'\s*(?:(\([0-9]+\))|([1-9]+)?(H)?([ABCDEF]+)(?:\{([0-9]+)\})?|([1-9]+))')
    while pos<len(text):
        if not text[pos:].strip():break
        match=token.match(text,pos)
        if match is None:raise ValueError(f'invalid command notation at {pos}')
        wait,path,charged,buttons,hold,axes=match.groups()
        pos=match.end();command+=1
        if wait:
            frames=int(wait[1:-1])
            if not 1<=frames<=10000:raise ValueError('bounded nonzero wait required')
            stages.append(Keys(5,'',frames,command));continue
        if axes:
            previous=None
            for d in axes:
                if d==previous:stages.append(Keys(5,'',release_frames,command))
                stages.append(Keys(int(d),'',tap_frames,command));previous=d
            continue
        path=path or '5'
        duration=int(hold) if hold else charge_frames if charged else tap_frames
        if not 1<=duration<=10000:raise ValueError('bounded nonzero hold required')
        # Repeated letters are repeated presses; explicit + is deliberately absent.
        # Simultaneous buttons use structured Keys instead of ambiguous AAA parsing.
        for index,button in enumerate(buttons):
            if index:command+=1
            directions=path if index==0 else path[-1]
            for index,d in enumerate(directions[:-1]):
                if index and d==directions[index-1]:stages.append(Keys(5,'',release_frames,command))
                stages.append(Keys(int(d),'',tap_frames,command))
            if len(directions)>1 and directions[-1]==directions[-2]:
                stages.append(Keys(5,'',release_frames,command))
            stages.append(Keys(int(directions[-1]),button,duration,command))
            stages.append(Keys(5,'',release_frames,command))
            if confirm:stages.append(WaitFor('accepted',timeout=120,retry=True))
    if sum(stage.frames for stage in stages if isinstance(stage,Keys))>100000:
        raise ValueError('command program too long')
    return tuple(stages)


class CommandPlayer:
    """Per-agent queue. next() submits exactly one frame; no per-move handler."""
    def __init__(self, program):
        self.queue=deque(program)
        if any(not isinstance(stage,(Keys,WaitFor)) for stage in self.queue):
            raise TypeError('command program contains an invalid stage')
        for stage in self.queue:
            if isinstance(stage,Keys) and (stage.direction not in DIRECTIONS
                    or type(stage.frames) is not int or not 1<=stage.frames<=10000
                    or any(button not in BUTTONS for button in stage.buttons)):
                raise ValueError('invalid key stage')
        normalized=[];serial=0;previous=None
        for stage in self.queue:
            if isinstance(stage,WaitFor):
                normalized.append(stage);previous=None;continue
            if stage.command!=previous:serial+=1
            previous=stage.command
            normalized.append(replace(stage,command=serial))
        self.queue=deque(normalized)
        self.current=None;self.remaining=0;self.facing=None;self.waited=0
        self.last_frame=None;self.segment=None
        self.command_id=None
        self.feedback=[];self.native_roots={};self.command_buttons={}
        self.issued={}

    def observe_native(self, events, seat):
        """Execution feedback only; never authorizes reward input attribution."""
        if seat not in (0,1):raise ValueError('two-player seat required')
        if self.command_id is None:return
        expected=self.command_buttons.get(self.command_id,())
        for event in events:
            if event.get('attacker_seat')!=seat:continue
            root=event.get('attack_instance')
            if (event['kind']=='ACTION_TRANSITION' and event['accepted']
                    and event.get('native_input_scope') and event.get('lineage')=='DIRECT'):
                current=event.get('native_inputs_before')
                buffered=event.get('buffered_inputs_before')
                released=event.get('released_inputs_before')
                if expected and any((current and current[b]>0) or (buffered and buffered[b]>0)
                        or (released and 0<released[b-2]<=2) for b in expected):
                    self.native_roots[root]=self.command_id
                    self.feedback.append(dict(kind='ACCEPTED',command_id=self.command_id))
            elif root in self.native_roots and event.get('hit_or_block') in ('HIT','BLOCK'):
                self.feedback.append(dict(kind=event['hit_or_block'],command_id=self.native_roots[root]))

    @property
    def done(self):return self.current is None and not self.queue

    def next(self, frame, facing, *, segment=0, state=None, events=()):
        if facing not in (-1,1):raise ValueError('facing must be -1 or 1')
        if self.last_frame is not None and (frame!=self.last_frame+1 or segment!=self.segment):
            raise RuntimeError('command player requires consecutive frames in one episode')
        self.last_frame=frame;self.segment=segment
        while self.current is None and self.queue:
            stage=self.queue[0]
            if isinstance(stage,WaitFor):
                state=state or {}
                if stage.condition in ('accepted','hit','block'):
                    ready=any(event.get('command_id')==self.command_id and
                        event.get('kind')==stage.condition.upper() for event in (*self.feedback,*events))
                elif stage.condition=='idle':ready=state.get('action') in (0,1,2,3,4,5)
                elif stage.condition=='action':ready=state.get('action')==stage.value
                else:
                    flags=state.get('frame_flags')
                    if flags is None:raise NotImplementedError('native frame flags required')
                    ready=bool(flags&0x20) and state.get('hitstop',0)==0
                if not ready:
                    self.waited+=1
                    if self.waited>stage.timeout:raise TimeoutError(f'chain gate {stage.condition} timed out')
                    if stage.retry:
                        issued=self.issued.get(self.command_id,())
                        if not issued:raise RuntimeError('acceptance retry has no submitted command')
                        return issued[(self.waited-1)%len(issued)]
                    return NEUTRAL
                self.queue.popleft();self.waited=0;continue
            self.current=self.queue.popleft();self.remaining=self.current.frames
            if self.command_id!=self.current.command:
                self.command_id=self.current.command;self.facing=facing
            if self.current.buttons:
                self.command_buttons[self.command_id]=tuple(BUTTONS[b] for b in self.current.buttons)
        if self.current is None:return NEUTRAL
        stage=self.current
        horizontal,vertical=DIRECTIONS[stage.direction]
        values=[horizontal*self.facing,vertical,0,0,0,0,0,0]
        for button in stage.buttons:values[BUTTONS[button]]=1
        self.remaining-=1
        self.issued.setdefault(self.command_id,[]).append(tuple(values))
        if not self.remaining:self.current=None
        return tuple(values)

    def action(self, *args, **kwargs):
        return encode_action(self.next(*args,**kwargs))

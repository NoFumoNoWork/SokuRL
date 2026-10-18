"""Approved opt-in join of existing native input/action events, not a new Hook.

Frame-end key state alone is NOT input consumption. Require matching native
input-handling scope(s) and retain the real event IDs. verified_sequence is an
adapter receipt ordinal, never an EXE/REP input-poll index.
"""
from collections import Counter

from sokurl_platform.command_input import CommandPlayer, compile_commands
from soku_rl.env.encoding import decode_action
from .continuous import ExecutionReceipt


def logical_keys(values):
    if values is None or len(values)!=8:
        raise ValueError('complete native logical key tuple required')
    return tuple((int(v>0)-int(v<0)) if i<2 else int(v>0) for i,v in enumerate(values))


class NativeTeacherFeedback:
    def __init__(self, interface, seat, segment, *, initial_facing):
        if seat not in (0,1):
            raise ValueError('absolute native seat required')
        self.interface, self.seat, self.segment = interface, seat, segment
        if initial_facing not in (-1,1):
            raise ValueError('actual reset facing required')
        self.previous_facing=initial_facing
        self.frame = 0
        self.verified_sequence = 0
        self.previous = (0,)*8
        self.counts = Counter()

    def observe(self, frame, events, submitted, teacher, current):
        if frame!=self.frame+1 or current.world['frame']!=frame:
            raise ValueError('one real simulation step per receipt required')
        if any(e['frame']!=frame or e['segment']!=self.segment for e in events):
            raise ValueError('native event frame/segment mismatch')
        self.frame = frame
        keys = decode_action(self.interface.command(submitted)).inputs
        own = [e for e in events if e.get('attacker_seat')==self.seat]
        samples = [e for e in own if e['kind']=='NATIVE_INPUT_SAMPLE']
        scopes = [e for e in own if e['kind']=='NATIVE_INPUT_SCOPE']
        verified = (len(samples)==1 and logical_keys(samples[0]['native_inputs_before'])==keys
                    and bool(scopes) and all(logical_keys(e['native_inputs_before'])==keys for e in scopes))
        reason = ('VERIFIED_NATIVE_INPUT_SCOPE' if verified else 'MISSING_OR_CONFLICTING_NATIVE_INPUT_EVIDENCE')
        if verified:
            self.verified_sequence += 1
        scope_ids = {e['native_input_scope'] for e in scopes}
        matching_plan = teacher.pending is not None and teacher.pending['candidate']==submitted
        accepted_buttons = []
        roots = []
        if verified and matching_plan:
            candidates = [b for i,b in enumerate(('A','B','C'),2) if keys[i] or self.previous[i]]
            if len(candidates)==1:
                b = candidates[0]
                index = {'A':2,'B':3,'C':4}[b]
                selected = []
                for e in own:
                    if e['kind']!='ACTION_TRANSITION' or e.get('native_input_scope') not in scope_ids:
                        continue
                    if type(e.get('attack_instance')) is not int or e['attack_instance']<=0:
                        continue
                    other = [j for j in (2,3,4) if j!=index]
                    native, buffered, released = (e.get('native_inputs_before'),
                        e.get('buffered_inputs_before'), e.get('released_inputs_before'))
                    if native is None or buffered is None or released is None:
                        continue
                    if any(native[j]>0 or buffered[j]>0 or 0<released[j-2]<=2 for j in other):
                        self.counts['ambiguous_acceptance'] += 1
                        continue
                    selected.append(e)
                # Reuse Platform's existing accepted DIRECT/input-scope rules;
                # next() only initializes this private observer,never writes game.
                observer = CommandPlayer(compile_commands('5'+b))
                observer.next(0, int(current.players[0]['dir']))
                observer.observe_native(selected,self.seat)
                if any(row['kind']=='ACCEPTED' for row in observer.feedback):
                    accepted_buttons = ['ACT_'+b]
                    roots = list(observer.native_roots)
        p = current.players[0]
        # Original main evaluates its own action/skill conditions. We only prove
        # a real parser reentry;never invent unavailable timers or call a
        # candidate movement mask a complete engine-legal action table.
        interrupted = bool(p['hitstop'] or p['hp']<=0 or any(
            e['kind']=='ACTION_TRANSITION' and e.get('native_input_scope') not in scope_ids
            for e in own))
        can_query = bool(verified and not p['hitstop'] and p['hp']>0)
        accepted = bool(accepted_buttons)
        receipt = ExecutionReceipt(frame-1, int(submitted), int(self.previous_facing),
            self.verified_sequence if verified else None, bool(verified), accepted,
            interrupted=interrupted,
            accepted_plan_id=teacher.serial if accepted else None,
            accepted_buttons=tuple(accepted_buttons))
        self.previous = keys
        self.previous_facing=int(p['dir'])
        self.counts[reason] += 1
        self.counts['accepted_receipts'] += int(accepted)
        evidence = dict(native_frame=frame,decision_frame=frame-1,reason=reason,
            verified_sequence=self.verified_sequence if verified else None,
            native_scope_ids=sorted(scope_ids),event_orders=[e['order'] for e in own],
            facing_before=receipt.facing,facing_after=int(p['dir']),
            accepted_roots=roots,boundary='OBSERVED_NATIVE_INPUT_HANDLING_SCOPE' if verified else 'NONE',
            boundary_semantics='SOURCE_MAIN_REENTRY_NOT_COMPLETE_MOVE_LEGALITY')
        return receipt,can_query,evidence

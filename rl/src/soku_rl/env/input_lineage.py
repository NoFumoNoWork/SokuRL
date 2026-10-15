"""Join native input-handling scopes to successful contact roots, not HUD hits."""
# th123_ai/source/motion.txt: public normal action names; not per-character handlers.
ACTION_BUTTONS = {**dict.fromkeys((300,301,302,303,304,305,306,307,308,309,311,320,321,322,323,330),2),
                  **dict.fromkeys((400,401,402,404,405,406,407,408,409),3),
                  **dict.fromkeys((410,411,412,414,415,416,418),4)}
# th123_ai/source/skill.txt B/C columns for default Reimu/Marisa skills.
SKILL_BUTTONS = {
    0:{**dict.fromkeys((500,502,520,522,540,560),(3,)),
       **dict.fromkeys((501,503,521,523,541,561),(4,))},
    1:{500:(3,4),**dict.fromkeys((520,540,542,560,562),(3,)),
       **dict.fromkeys((521,541,543,561,563),(4,))},
}


class InputLineage:
    """Generic ledger; action-table inference is an explicit diagnostic only."""
    def __init__(self, *, diagnostic_action_mapping=False, accepted_scope_decisions=False,
                 accepted_action_roots=False):
        if sum(map(bool,(diagnostic_action_mapping,accepted_scope_decisions,accepted_action_roots)))>1:
            raise ValueError('select one lineage evidence model')
        self.diagnostic_action_mapping=diagnostic_action_mapping
        self.accepted_scope_decisions=accepted_scope_decisions
        self.accepted_action_roots=accepted_action_roots
        self.segment=None

    def reset(self, segment):
        self.segment=segment
        self.held=[[False]*8 for _ in range(2)]
        self.press=[[None]*8 for _ in range(2)]
        self.buffer_press=[[None]*8 for _ in range(2)]
        self.buffered=[[False]*8 for _ in range(2)]
        self.serial=0
        self.decision_frame=[None,None]
        self.decision_token=[None,None]
        self.scopes={}
        self.actions={}
        self.combo=[[],[]]
        self.combo_roots=[{},{}]
        self.combo_serial=[0,0]
        self.input_frame=[None,None]
        self.characters=None

    def observe(self, events, characters=None):
        output=[]
        for source in events:
            event=source.copy()
            if event['segment']!=self.segment:self.reset(event['segment'])
            if characters is not None:
                if self.characters is not None and self.characters!=tuple(characters):
                    raise RuntimeError('characters changed without a new input epoch')
                self.characters=tuple(characters)
            seat=event['attacker_seat']
            if event['kind'] in ('NATIVE_INPUT_SCOPE','NATIVE_INPUT_SAMPLE'):
                if self.accepted_action_roots:
                    output.append(event)
                    continue
                keys=event['native_inputs_before']
                buffered=event.get('buffered_inputs_before')
                gap=self.input_frame[seat] is not None and event['frame']>self.input_frame[seat]+1
                for button in range(2,8):
                    down=keys[button]>0
                    if gap:
                        self.press[seat][button]=None
                        self.buffer_press[seat][button]=None
                    if down and not self.held[seat][button]:
                        if self.accepted_scope_decisions:
                            if self.decision_frame[seat]!=event['frame']:
                                self.serial+=1
                                self.decision_frame[seat]=event['frame']
                                self.decision_token[seat]=self.serial
                            self.press[seat][button]=self.decision_token[seat]
                        else:
                            self.serial+=1
                            self.press[seat][button]=self.serial
                    self.held[seat][button]=down
                    buffered_down=bool(buffered and buffered[button]>0)
                    if not buffered_down:
                        self.buffer_press[seat][button]=None
                    elif not self.buffered[seat][button]:
                        token=self.press[seat][button]
                        self.buffer_press[seat][button]=frozenset((token,)) if token is not None else None
                    elif down:
                        possible=self.buffer_press[seat][button]
                        token=self.press[seat][button]
                        self.buffer_press[seat][button]=possible | {token} if possible is not None and token is not None else None
                    self.buffered[seat][button]=buffered_down
                self.input_frame[seat]=event['frame']
                if event['kind']=='NATIVE_INPUT_SCOPE':
                    self.scopes[(seat,event['native_input_scope'])]=(tuple(self.press[seat]),tuple(self.buffer_press[seat]))
            elif event['kind']=='ACTION_TRANSITION':
                if self.accepted_action_roots:
                    # The accepted action operation is the identity, not a
                    # reconstruction of which buffered button it consumed.
                    root=event.get('attack_instance')
                    accepted=(event.get('accepted') and event.get('lineage')=='DIRECT'
                              and event.get('native_input_scope') and root is not None)
                    self.actions[(seat,root)]=frozenset((root,)) if accepted else None
                    event['attack_start_evidence']='NATIVE_INPUT_ACCEPTED_ACTION_INSTANCE' if accepted else 'UNKNOWN'
                    event['independent_action_instance']=root if accepted else None
                    output.append(event)
                    continue
                buttons=()
                if self.diagnostic_action_mapping:
                    button=ACTION_BUTTONS.get(event['action_after'])
                    buttons=(button,) if button is not None else SKILL_BUTTONS.get(
                        self.characters[seat],{}).get(event['action_after'],()) if self.characters else ()
                scope=self.scopes.get((seat,event.get('native_input_scope')))
                token=None
                if self.accepted_scope_decisions:
                    # Generic ABC decision candidates in a real native input scope.
                    # This is NOT a claim of instruction-level consumed-key tracing.
                    buttons=(2,3,4)
                if ((self.diagnostic_action_mapping or self.accepted_scope_decisions) and buttons and scope
                        and event['accepted'] and event['lineage']=='DIRECT'):
                    current=event['native_inputs_before']
                    buffered=event.get('buffered_inputs_before')
                    released=event.get('released_inputs_before')
                    candidates=[]
                    for button in buttons:
                        direct=scope[0][button]
                        if current[button]>0:candidates.append(frozenset((direct,)) if direct is not None else None)
                        if buffered and buffered[button]>0:candidates.append(scope[1][button])
                        if released and 0<released[button-2]<=2:
                            candidates.append(frozenset((direct,)) if current[button]==0 and direct is not None else None)
                    if candidates and None not in candidates:
                        token=frozenset().union(*candidates)
                consumed=event.get('accepted_input_candidates')
                if consumed is not None:
                    if (event.get('input_acceptance_evidence')!='NATIVE_INPUT_CONSUMPTION'
                            or not event['accepted'] or event['lineage']!='DIRECT'
                            or not consumed or any(type(value) is not int or value<=0 for value in consumed)):
                        raise ValueError('invalid native input consumption evidence')
                    token=frozenset(consumed)
                self.actions[(seat,event['attack_instance'])]=token
                previous_evidence=event.get('input_acceptance_evidence')
                event['input_acceptance_evidence']=(previous_evidence if previous_evidence not in (None,'UNKNOWN') else None) or (
                    'NATIVE_ACCEPTED_SCOPE_DECISION_CANDIDATES' if token and self.accepted_scope_decisions else 'UNKNOWN')
                event['independent_input_id']=next(iter(token)) if token and len(token)==1 else None
                event['input_candidates']=tuple(sorted(token)) if token else None
            elif event['kind']=='HIT_RESOLUTION' and event['hit_or_block']=='HIT' and event['hp_after']<event['hp_before']:
                token=self.actions.get((seat,event['attack_instance']))
                root=event['attack_instance']
                if self.accepted_action_roots and event.get('lineage') not in ('DIRECT','CHILD'):
                    root=token=None
                count=event.get('combo_count_at_damage')
                event.update(independent_input_segment=None,independent_attack_index=None,
                             new_independent_attack=False)
                if count==0:
                    self.combo_serial[seat]+=1
                    self.combo[seat]=[token]
                    self.combo_roots[seat]={root:1} if root is not None else {}
                    event['independent_input_segment']=False
                    event['independent_attack_index']=1
                    event['new_independent_attack']=True
                elif root is not None and root in self.combo_roots[seat]:
                    index=self.combo_roots[seat][root]
                    event['independent_attack_index']=index
                    event['independent_input_segment']=index>1
                elif count is not None and token is not None and self.combo[seat] and None not in self.combo[seat]:
                    index=None
                    used=frozenset().union(*self.combo[seat])
                    if token.isdisjoint(used):
                        self.combo[seat].append(token)
                        index=len(self.combo[seat])
                        event['new_independent_attack']=True
                    elif len(token)==1 and token in self.combo[seat]:
                        index=self.combo[seat].index(token)+1
                    if index is not None:
                        self.combo_roots[seat][root]=index
                        event['independent_attack_index']=index
                        event['independent_input_segment']=index>1
                event['independent_input_id']=next(iter(token)) if token and len(token)==1 and not self.accepted_action_roots else None
                if self.accepted_action_roots:
                    event['independent_action_instance']=next(iter(token)) if token else None
                    event['attack_root_evidence']='NATIVE_ACTION_SPAWN_ROOT' if root is not None else 'UNKNOWN'
                    event['combo_evidence']='NATIVE_DAMAGE_TIME_COMBO_COUNT' if count is not None else 'UNKNOWN'
                event['native_combo_id']=self.combo_serial[seat]
                if self.accepted_action_roots and event['independent_input_segment'] is None:
                    action=(seat,event.get('attack_instance'))
                    if count is None:reason='MISSING_NATIVE_COMBO_COUNT'
                    elif event.get('lineage') not in ('DIRECT','CHILD'):reason='UNPROVEN_SPAWN_PROVENANCE'
                    elif event.get('attack_instance') is None:reason='MISSING_ATTACK_ROOT'
                    elif action not in self.actions:reason='ACTION_INSTANCE_NOT_OBSERVED'
                    elif self.actions[action] is None:reason='ACTION_NOT_PROVEN_AS_INPUT_ACCEPTED'
                    elif not self.combo[seat] or None in self.combo[seat]:reason='EARLIER_COMBO_ACTION_UNPROVEN'
                    else:reason='UNRESOLVED_ACTION_ROOT_RELATION'
                    event['independent_segment_unknown_reason']=reason
                event['input_candidates']=tuple(sorted(token)) if token and not self.accepted_action_roots else None
                event['input_lineage_evidence']=(
                    'NATIVE_INPUT_ACCEPTED_ACTION_INSTANCE' if self.accepted_action_roots else
                    'DIAGNOSTIC_ACTION_MAPPING' if self.diagnostic_action_mapping else
                    'NATIVE_ACCEPTED_SCOPE_DECISION_CANDIDATES' if self.accepted_scope_decisions else
                    'NATIVE_INPUT_CONSUMPTION') if token else 'UNKNOWN'
            output.append(event)
        return output

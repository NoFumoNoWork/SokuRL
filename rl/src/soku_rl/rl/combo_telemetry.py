"""Observe existing native combo IDs and accepted roots; never infer recovery."""
from collections import Counter,deque


class ComboTelemetry:
    def __init__(self):
        self.active={}
        self.sources={}
        self.episodes={}
        self.reset_batch()

    def reset_batch(self):
        self.counts=dict(hits=0,counter_openers=0,combo_extensions=0,unknown_lineage_hits=0,
                         counter_damage=0.,unknown_counter_hits=0,
                         hits_taken=0,counter_taken=0,counter_damage_taken=0.,
                         blocked_attacks=0,blocks_made=0,
                         unresolved_hits=0,available_transitions=0,unavailable_transitions=0)
        self.closed=[]
        self.touched={}
        self.unknown_events=[]

    def _close(self,slot,reason):
        value=self.active.pop(slot,None)
        if value is not None:
            self.closed.append(dict(value,reason=reason,
                complete=not value['left_censored'] and reason in ('NEW_NATIVE_COMBO','EPISODE_END')))

    def observe(self,slot,info,done,learner_steps=None):
        if 'native_combat_events' not in info:
            self.counts['unavailable_transitions']+=1
            return
        self.counts['available_transitions']+=1
        seat,episode=info['learner_seat'],info['episode_id']
        if self.episodes.get(slot)!=episode:
            self.sources[slot]=deque(maxlen=256)
            self.episodes[slot]=episode
        current=self.active.get(slot)
        if current is not None and current['episode_id']!=episode:
            self._close(slot,'RESET_CENSORED')
        for event in info['native_combat_events']:
            if event['kind'] in ('ACTION_TRANSITION','OBJECT_CREATED') and event['attacker_seat']==seat:
                fields=('kind','frame','order','segment','source_address','source_instance','attack_instance',
                        'lineage','action','sequence','action_before','action_after','accepted',
                        'native_input_scope','caller','parent_address','attack_start_evidence')
                self.sources[slot].append({key:event[key] for key in fields if key in event})
            if event['kind'] in ('BLOCK_HIGH_RESOLUTION','BLOCK_LOW_RESOLUTION') and event['hit_or_block']=='BLOCK':
                self.counts['blocked_attacks' if event['attacker_seat']==seat else 'blocks_made']+=1
            if (event['kind']!='HIT_RESOLUTION' or event['hit_or_block']!='HIT'
                    or event['hp_after']>=event['hp_before']):
                continue
            if event['attacker_seat']!=seat:
                self.counts['hits_taken']+=1
                if event.get('counter_hit_first') is True:
                    self.counts['counter_taken']+=1
                    self.counts['counter_damage_taken']+=event['hp_before']-event['hp_after']
                continue
            self.counts['hits']+=1
            self.counts['combo_extensions']+=int(event.get('new_independent_attack') is True
                and event.get('independent_input_segment') is True)
            self.counts['unknown_counter_hits']+=int(type(event.get('counter_hit_first')) is not bool)
            unknown=type(event.get('independent_input_segment')) is not bool
            self.counts['unknown_lineage_hits']+=int(unknown)
            if unknown:
                root=event.get('attack_instance')
                sources=[row for row in self.sources[slot] if
                    (root is not None and row.get('attack_instance')==root) or
                    (row['kind']=='OBJECT_CREATED' and row.get('source_instance')==event.get('source_instance'))]
                parents={row.get('parent_address') for row in sources if row.get('parent_address')}
                sources.extend(row for row in self.sources[slot] if
                    row['kind']=='OBJECT_CREATED' and row.get('source_address') in parents)
                self.unknown_events.append(dict(event=event.copy(),source_context=sources,
                    context_limit=256,slot=slot,episode_id=episode,
                    limitation='Observed context only; not proof that owner/current action identifies attack root.'))
            identity=event.get('native_combo_id')
            count=event.get('combo_count_after')
            if identity is None or count is None:
                self.counts['unresolved_hits']+=1
                if event.get('counter_hit_first') is True:
                    self.counts['counter_openers']+=1
                    self.counts['counter_damage']+=event['hp_before']-event['hp_after']
                continue
            key=(episode,event['segment'],seat,identity)
            current=self.active.get(slot)
            if current is not None and current['key']!=key:
                self._close(slot,'NEW_NATIVE_COMBO')
                current=None
            if current is None:
                current=dict(key=key,slot=slot,episode_id=episode,seat=seat,
                    first_hit_frame=event['frame'],last_hit_frame=event['frame'],
                    first_seen_steps=learner_steps,last_hit_steps=learner_steps,
                    native_hits=0,independent_segments_lower_bound=0,damage=0.,
                    counter_opened=False,unknown_lineage_hits=0,
                    left_censored=event.get('combo_count_at_damage')!=0)
                self.active[slot]=current
            current['native_hits']=max(current['native_hits'],int(count))
            current['independent_segments_lower_bound']=max(current['independent_segments_lower_bound'],
                event.get('independent_attack_index') or 0)
            current['damage']+=event['hp_before']-event['hp_after']
            current['last_hit_frame']=event['frame']
            current['last_hit_steps']=learner_steps
            current['unknown_lineage_hits']+=int(unknown)
            current['last_damage_rate']=event.get('damage_rate')
            if event.get('counter_hit_first') is True and not current['counter_opened']:
                current['counter_opened']=True
                self.counts['counter_openers']+=1
                self.counts['counter_damage']+=event['hp_before']-event['hp_after']
            self.touched[key]=current.copy()
        if done:self._close(slot,'EPISODE_END')

    def snapshot(self):
        complete=[value for value in self.closed if value['complete']]
        def histogram(key):return dict(sorted(Counter(v[key] for v in complete).items()))
        return self.counts | dict(completed_combos=len(complete),
            native_hit_length_histogram=histogram('native_hits'),
            independent_segment_lower_bound_histogram=histogram('independent_segments_lower_bound'),
            mean_completed_hits=sum(v['native_hits'] for v in complete)/len(complete) if complete else None,
            mean_completed_segments=sum(v['independent_segments_lower_bound'] for v in complete)/len(complete) if complete else None,
            max_observed_hits=max((v['native_hits'] for v in self.touched.values()),default=0),
            max_observed_segments=max((v['independent_segments_lower_bound'] for v in self.touched.values()),default=0),
            completed_damage=sum(v['damage'] for v in complete),
            ch_opened_completed_combos=sum(v['counter_opened'] for v in complete),
            completed_with_unknown_lineage=sum(v['unknown_lineage_hits']>0 for v in complete),
            fully_known_segment_combos=sum(v['unknown_lineage_hits']==0 for v in complete),
            unknown_events=self.unknown_events.copy(),
            open_combos=[dict(value) for value in self.active.values()],
            closed_combos=[dict(value) for value in self.closed],
            boundary='Only next native combo or episode end confirms closure; rollout boundaries do not.')

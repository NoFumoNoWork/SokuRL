"""Offline decision evidence, not a second native reader or move interpreter."""
from dataclasses import asdict

import numpy as np

from sokurl_platform.command_input import compile_input_frames
from sokurl_platform.replay_inputs import decode_keys
from soku_rl.env.input_lineage import InputLineage


def input_program(words):
    """Absolute logical inputs; execute with facing=1, never invent releases."""
    return [asdict(stage) for stage in compile_input_frames(
        tuple(decode_keys(int(word)).values()) for word in words)]


def weather_eligible(weather, *, cards_used=False, skills_default=True):
    # Public Platform constants, not locally guessed weather IDs.
    from sokurl_platform.bridge_shared import WEATHER_TYPHOON
    return not cards_used and skills_default and int(weather) != WEATHER_TYPHOON


def extract_chains(arrays, records, seat):
    """Acceptance-time labels and movement proxies, with future outcomes separate.

    Only original accepted rows authorize samples. Action IDs are native labels,
    NOT the policy's 90 primitive keys. No inference of command onset/legality.
    """
    n = len(arrays['actions'])
    frames = arrays['consumption_frames']
    if (seat not in (0, 1) or len(records) != n or
            not np.all(arrays['seats'] == seat) or
            not np.array_equal(frames, arrays['observation_frames'] + 1) or
            (n > 1 and not np.all(np.diff(frames) == 1))):
        raise ValueError('dense causal single-seat record required')
    valid = (arrays['supervised'] & arrays['compatible'] & ~arrays['boundary'] &
             arrays['raw_words_present'])
    ledger = InputLineage(accepted_action_roots=True)
    decisions, movements, transitions, contacts, objects = [], [], [], [], []
    combo_groups, roots = {}, {}
    movement = None
    for row, record in enumerate(records):
        frame = int(frames[row])
        if record['frame'] != frame:
            raise ValueError('event/observation frame mismatch')
        events = record['native_events']
        if any(e['frame'] != frame for e in events):
            raise ValueError('wrong-frame native event')
        order = [(e['segment'], e['frame'], e['order']) for e in events]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError('unordered or duplicated native events')
        decoded = ledger.observe(events)
        own = [e for e in decoded if e['attacker_seat'] == seat]
        if not valid[row]:
            movement = None
            continue
        accepted = []
        for event in own:
            key = (event['segment'], event.get('attack_instance'))
            evidence = dict(frame=frame, order=event['order'], segment=event['segment'])
            if event['kind'] == 'ACTION_TRANSITION':
                is_decision = event.get('independent_action_instance') is not None
                transitions.append(dict(**evidence, action_before=event['action_before'],
                    action_after=event['action_after'], accepted=event['accepted'],
                    classification='INPUT_ACCEPTED_OPERATION' if is_decision else 'NOT_PROVEN_INPUT_DECISION'))
                if is_decision:
                    if key in roots:
                        raise ValueError('duplicate accepted action instance')
                    item = dict(**evidence, id=len(decisions), observation_row=row,
                        observation_frame=int(arrays['observation_frames'][row]),
                        action_id=event['action_after'], attack_instance=event['attack_instance'],
                        kind='ACCEPTED_ACTION', evidence='NATIVE_INPUT_ACCEPTED_ACTION_INSTANCE',
                        timing='S[f-1] before acceptance in f;NOT proven command onset',
                        command_name=None, command_onset=None, next_decision=None,
                        command_combination_before=event.get('command_combination_before'),
                        movement_combination_before=event.get('movement_combination_before'),
                        input_history_status='May already contain part of this command',
                        outcomes=dict(hits=0, blocks=0, damage=0, objects=0))
                    roots[key] = item
                    decisions.append(item)
                    accepted.append(item)
            elif event['kind'] == 'OBJECT_CREATED':
                root = roots.get(key) if event.get('lineage') in ('DIRECT', 'CHILD') else None
                objects.append(dict(**evidence, action_id=event['action'],
                    attack_instance=event.get('attack_instance'), lineage=event['lineage'],
                    source_instance=event.get('source_instance'),
                    decision_id=root['id'] if root else None))
                if root:
                    root['outcomes']['objects'] += 1
            elif event.get('hit_or_block') in ('HIT', 'BLOCK'):
                root = roots.get(key) if event.get('lineage') in ('DIRECT', 'CHILD') else None
                damage = max(0, event['hp_before'] - event['hp_after'])
                contact = dict(**evidence, kind=event['hit_or_block'], damage=damage,
                    attack_instance=event.get('attack_instance'), lineage=event.get('lineage'),
                    combo_id=event.get('native_combo_id') if event.get('combo_count_at_damage') is not None else None,
                    combo_count_after=event['combo_count_after'],
                    independent_attack_index=event.get('independent_attack_index'),
                    decision_id=root['id'] if root else None,
                    attribution='CONFIRMED_ROOT' if root else 'UNKNOWN_ROOT')
                contacts.append(contact)
                if root:
                    root['outcomes']['hits' if event['hit_or_block'] == 'HIT' else 'blocks'] += 1
                    root['outcomes']['damage'] += damage
                if event['hit_or_block'] == 'HIT' and damage and contact['combo_id'] is not None:
                    combo_groups.setdefault((event['segment'], contact['combo_id']), []).append(contact)
        for item in accepted:
            item['same_frame_acceptances'] = len(accepted)
        # Movement is submitted input evidence, NOT proof that a dash/jump fired.
        logical = tuple(decode_keys(int(arrays['raw_words'][row, seat])).values())
        if not any(logical[2:5]) and not any(logical[6:]) and not accepted:
            signature = (logical[0], logical[1], logical[5])
            if movement is not None and movement['end_row'] == row - 1 and movement['logical'] == signature:
                movement.update(end_row=row, end_frame=frame)
            else:
                movement = dict(id=len(movements), start_row=row, end_row=row,
                    start_frame=frame, end_frame=frame, logical=signature,
                    evidence='RAW_LOGICAL_INPUT_RUN', accepted_movement='UNKNOWN',
                    kind='RELEASE' if signature == (0, 0, 0) else 'MOVEMENT_INPUT')
                movements.append(movement)
        else:
            movement = None
    for index, item in enumerate(decisions[:-1]):
        following = decisions[index + 1]
        a, b = item['observation_row'], following['observation_row']
        if item['segment'] == following['segment'] and valid[a:b+1].all():
            item['next_decision'] = dict(id=following['id'], action_id=following['action_id'],
                delta_frames=following['frame'] - item['frame'],
                evidence='OBSERVED_SUCCESSION_NOT_PROVEN_CANCEL')
    combos = []
    for (segment, serial), hits in combo_groups.items():
        order = list(dict.fromkeys(h['decision_id'] for h in hits if h['decision_id'] is not None))
        start_frame = min([hits[0]['frame'],*[decisions[i]['frame'] for i in order]])
        end_frame = hits[-1]['frame']
        span_rows = np.flatnonzero((frames>=start_frame)&(frames<=end_frame))
        continuous = bool(len(span_rows) and valid[span_rows].all())
        # Behavior links need not cause damage: jump/flight/release may connect hits.
        steps = [dict(kind='ACCEPTED_OPERATION',id=d['id'],frame=d['frame'],order=d['order'],
                      action_id=d['action_id'],input_acceptance_confirmed=True,
                      contributed_contact=d['id'] in order)
                 for d in decisions if d['segment']==segment and start_frame<=d['frame']<=end_frame]
        steps.extend(dict(kind=m['kind'],id=m['id'],frame=max(m['start_frame'],start_frame),
                          end_frame=min(m['end_frame'],end_frame),order=0,
                          logical=m['logical'],input_acceptance_confirmed=False,
                          contributed_contact=False)
                     for m in movements if m['start_frame']<=end_frame and m['end_frame']>=start_frame)
        steps.sort(key=lambda s:(s['frame'],s['order'],s['id']))
        combos.append(dict(segment=segment, native_combo_id=serial, hits=len(hits),
            first_frame=hits[0]['frame'], last_frame=hits[-1]['frame'],
            damage=sum(h['damage'] for h in hits), decision_ids=order,
            action_ids=[decisions[i]['action_id'] for i in order],
            independent_actions=len(order), unknown_root_hits=sum(h['decision_id'] is None for h in hits),
            source_operation_semantics='Distinct accepted operation roots;NOT necessarily attack moves',
            behavior_start_frame=start_frame,behavior_chain_eligible=continuous,
            behavior_steps=steps,
            behavior_evidence='OBSERVED_CHRONOLOGY_NOT_PROVEN_NECESSITY_OR_CANCEL',
            context_transitions=[t for t in transitions if t['segment']==segment and
                start_frame<=t['frame']<=end_frame],
            damage_sources_in_p0_attack_range=(bool(order) and not any(h['decision_id'] is None for h in hits)
                and all(300<=decisions[i]['action_id']<600 for i in order)),
            non_attack_root_actions=[decisions[i]['action_id'] for i in order
                if not 300<=decisions[i]['action_id']<600],
            attack_range_evidence='P0 range diagnostic ONLY;never excludes movement from behavior supervision',
            complete_combo_boundary='NOT_PROVEN_AT_PREFIX_END', contacts=hits))
    timeline = [dict(kind='ACCEPTED_ACTION', id=d['id'], frame=d['frame'],
                     observation_row=d['observation_row']) for d in decisions]
    timeline.extend(dict(kind=m['kind'], id=m['id'], frame=m['start_frame'],
                         observation_row=m['start_row']) for m in movements)
    timeline.sort(key=lambda item: (item['frame'],item['kind'],item['id']))
    for current, following in zip(timeline,timeline[1:]):
        a,b=current['observation_row'],following['observation_row']
        current['next'] = dict(kind=following['kind'],id=following['id'],
            delta_frames=following['frame']-current['frame']) if valid[a:b+1].all() else None
    return dict(decisions=decisions, movements=movements, transitions=transitions,
                combos=combos, contacts=contacts, objects=objects, timeline=timeline)


def decision_examples(arrays, chains):
    """One supervised row per operation/run, rather than every held-input frame.

    Outcome fields/next actions never enter observations. This is an offline
    acceptance/movement dataset, not a ready-to-deploy 90-key PPO policy.
    """
    items = [(d['observation_row'], 1, d['action_id'], (0, 0, 0),
              d['same_frame_acceptances'] == 1, d['id']) for d in chains['decisions']]
    items.extend((m['start_row'], 2, -1, m['logical'], True, m['id']) for m in chains['movements'])
    items.sort(key=lambda v: (v[0], v[1], v[5]))
    rows = np.array([i[0] for i in items], dtype=np.int64)
    return dict(observations=arrays['observations'][rows], observation_rows=rows,
        observation_frames=arrays['observation_frames'][rows],
        kind=np.array([i[1] for i in items], dtype=np.int8),
        native_action_ids=np.array([i[2] for i in items], dtype=np.int32),
        movement=np.array([i[3] for i in items], dtype=np.int8).reshape(-1, 3),
        supervised=np.array([i[4] for i in items], dtype=bool),
        evidence_ids=np.array([i[5] for i in items], dtype=np.int64))

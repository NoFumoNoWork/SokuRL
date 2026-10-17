"""Timestamped public feedback, separate from retrospective supervision strata."""
import numpy as np

from soku_rl.replay.hierarchical import PublicEventHistory


FEEDBACK = {'HIT': (5, 1.), 'BLOCK': (6, 1.), 'WHIFF': (7, 1.),
            'CANCEL': (7, 2.), 'RECOVERY': (7, 3.)}


class SemanticEventHistory(PublicEventHistory):
    def observe(self, state, frame, last_own_input=None, *, feedback=()):
        changed = super().observe(state, frame, last_own_input)
        for item in feedback:
            if (item['available_frame'] != frame or item['occurred_frame'] > frame
                    or item['evidence'] != 'CONFIRMED' or not item['source']):
                raise ValueError('feedback must be confirmed and available at current observation')
            kind, result = FEEDBACK[item['kind']]
            owner = item['owner']
            if owner not in (0, 1):
                raise ValueError('relative owner required')
            action = int(state[owner * 10 + 5])
            self.events.append((kind, action, owner, frame, 0, result, np.asarray(state).copy()))
            changed = True
        return changed


def contact_feedback(events, seat, frame):
    """Use existing decoded HIT/BLOCK facts; never interpret raw EXE fields."""
    output = []
    seen = set()
    for event in events:
        if event.get('hit_or_block') not in ('HIT', 'BLOCK'):
            continue
        if event['frame'] > frame:
            raise ValueError('future native feedback')
        key = (event['segment'], event['frame'], event['order'])
        if key in seen:
            raise ValueError('duplicate feedback event')
        seen.add(key)
        output.append(dict(kind=event['hit_or_block'], owner=int(event['attacker_seat'] != seat),
            occurred_frame=event['frame'], available_frame=frame, evidence='CONFIRMED',
            source='Platform.' + event['kind']))
    return output


def event_strata(data, audit, chains, incoming_frames):
    """Overlapping label roles; observed links are NOT native cancel proof."""
    n = len(audit['samples'])
    strata = np.zeros((n, 4), bool)
    chain_ids = np.full(n, -1, np.int64)
    linked_rows = set()
    for d in chains['decisions']:
        following = d.get('next_decision')
        if following:
            linked_rows.add(chains['decisions'][following['id']]['observation_row'])
    for i, sample in enumerate(audit['samples']):
        selection = sample['selection']
        frame = sample['observation_frame']
        row = sample['row']
        known = bool(data['target_action_mask'][i])
        is_attack = selection is not None and selection['acceptance'] is not None
        strata[i, 0] = known and not is_attack
        strata[i, 1] = known and is_attack
        acceptance = selection['acceptance'] if selection else None
        strata[i, 2] = known and acceptance in linked_rows
        # Passive feedback is context/termination supervision, never an attack label.
        strata[i, 3] = frame in incoming_frames or 50 <= int(data['state'][i, 5]) < 200
        for ci, combo in enumerate(chains.get('combos', [])):
            if (combo['behavior_chain_eligible'] and combo['behavior_start_frame'] <= frame
                    <= combo['last_frame']):
                chain_ids[i] = ci
                if known:
                    strata[i, 2] = True
                break
        if not strata[i].any() and (data['target_termination_mask'][i]
                                   or data['target_sequence_mask'][i].any()):
            strata[i, 0] = True  # Known continuation at a public context event.
    usable = (data['target_action_mask'] | data['target_termination_mask'] |
              data['target_sequence_mask'].any(-1))
    return strata & usable[:, None], chain_ids

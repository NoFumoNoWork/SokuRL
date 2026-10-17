"""Single-command control and native-event-only V3-C reward adjustments."""
from dataclasses import dataclass
from math import isfinite

from sokurl_platform.command_input import CommandPlayer, Keys, compile_commands
from soku_rl.env.encoding import encode_action


def vocabulary():
    """Single moves, not preset combos. Timing specifies keys, not acceptance."""
    programs = {'neutral': (Keys(5, '', 1, 1),)}
    programs.update({name: (Keys(direction, '', 6, 1),) for name, direction in
                     [('forward', 6), ('back_guard', 4), ('crouch_guard', 1)]})
    for text in ('7', '8', '9', '7D', '8D', '9D', '6D', '44', '66',
                 '5A', '2A', '3A', '6A', '5B', '2B', '6B', '5C', '2C', '6C',
                 '236B', '236C', '623B', '623C', '214B', '214C', '22B', '22C',
                 '2B{40}', '5C{40}'):
        programs[text] = compile_commands(text)
    return tuple(programs.items())


@dataclass(frozen=True)
class V3RewardConfig:
    back_jump_hp: float = 40.
    back_jump_cap_hp: float = 240.
    back_d_hp: float = 20.
    reset_forward_frames: int = 16
    reset_forward_distance: float = 32.

    def __post_init__(self):
        if (any(not isfinite(v) or v < 0 for v in self.__dict__.values())
                or type(self.reset_forward_frames) is not int or self.reset_forward_frames < 2):
            raise ValueError('finite nonnegative costs and sustained reset required')


class V3Ledger:
    """Layer over unmodified V2/zero combo_beta; native HP and CH stay intact."""
    def __init__(self, config=V3RewardConfig()):
        self.config = config
        self.streak = [0, 0]
        self.progress = [0., 0.]
        self.forward = [0, 0]
        self.combo = [None, None]
        self.ordinal = [0, 0]
        self.attack_roots = [set(), set()]
        self.hit_roots = [set(), set()]
        self.totals = [dict(back_jumps=0, max_back_jump_streak=0, back_d=0,
                            back_jump_cost=0., back_d_cost=0., combo_hit_scaling=0.,
                            unknown_hit_ordinals=0, hits=0, accepted_actions=0,
                            attack_accepts=0, hit_attack_roots=0, accepted_attack_roots=0,
                            hits_by_source={}, hit_ordinals={}) for _ in range(2)]

    def step(self, events, before, after):
        parts = [dict(back_jump=0., back_d=0., combo_hit_scaling=0.) for _ in range(2)]
        c = self.config
        movement_entries = set()
        for seat in (0, 1):
            toward = (after[seat]['x'] - before[seat]['x']) * before[seat]['facing']
            if toward > 0 and after[seat]['action'] == 4:
                self.forward[seat] += 1
                self.progress[seat] += toward
                if self.forward[seat] >= c.reset_forward_frames and self.progress[seat] >= c.reset_forward_distance:
                    self.streak[seat] = 0
            else:
                self.forward[seat] = 0
                self.progress[seat] = 0.
        for event in events:
            seat = event['attacker_seat']
            row = self.totals[seat]
            if event['kind'] == 'ACTION_TRANSITION':
                if not (event.get('accepted') and event.get('native_input_scope')
                        and event.get('lineage') == 'DIRECT'):
                    continue
                row['accepted_actions'] += 1
                action = event['action_after']
                inputs = event.get('native_inputs_before')
                if 300 <= action < 700 and inputs and any(inputs[i] > 0 for i in (2, 3, 4)):
                    row['attack_accepts'] += 1
                    if event.get('attack_instance') is not None:
                        self.attack_roots[seat].add(event['attack_instance'])
                        row['accepted_attack_roots'] = len(self.attack_roots[seat])
                    self.streak[seat] = 0
                if (event['action_before'] == action or 50 <= event['action_before'] < 200
                        or not inputs):
                    continue
                # SokuLib Action.hpp:8/210 backward jump;201/203 backward dash.
                #214 (free flight) is NOT inferred to be backward from keys.
                if action in (8, 210):
                    movement_entries.add((seat, 'back_jump'))
                if action in (201, 203):
                    movement_entries.add((seat, 'back_d'))
            if event['kind'] not in ('HIT_RESOLUTION', 'BLOCK_HIGH_RESOLUTION', 'BLOCK_LOW_RESOLUTION'):
                continue
            if event.get('hit_or_block') in ('HIT', 'BLOCK'):
                self.streak = [0, 0]
            damage = max(0, event['hp_before'] - event['hp_after'])
            if not damage or event.get('hit_or_block') != 'HIT':
                continue
            ordinal, combo = event.get('combo_count_after'), event.get('native_combo_id')
            reliable = (type(ordinal) is int and ordinal > 0 and type(combo) is int and combo > 0
                        and (combo != self.combo[seat] or ordinal > self.ordinal[seat]))
            if reliable:
                factor = .5 if ordinal == 1 else 1 + .2 * (ordinal - 2)
                self.combo[seat], self.ordinal[seat] = combo, ordinal
                key = str(ordinal)
                row['hit_ordinals'][key] = row['hit_ordinals'].get(key, 0) + 1
            else:
                factor = .5
                row['unknown_hit_ordinals'] += 1
            delta = damage * (factor - 1)
            parts[seat]['combo_hit_scaling'] += delta
            row['combo_hit_scaling'] += delta
            row['hits'] += 1
            if event.get('attack_instance') in self.attack_roots[seat]:
                self.hit_roots[seat].add(event['attack_instance'])
                row['hit_attack_roots'] = len(self.hit_roots[seat])
            source = str(event['source_action'])
            row['hits_by_source'][source] = row['hits_by_source'].get(source, 0) + 1
        # Common movement can assign native actionId without calling setAction.
        # A real frame-boundary entry is evidence; held keys alone are not.
        for seat in (0, 1):
            old, new = before[seat], after[seat]
            if (old['action'] == new['action'] or 50 <= old['action'] < 200
                    or old.get('hitstop', 0) > 0 or old.get('untech', 0) > 0):
                continue
            if new['action'] in (8, 210):
                movement_entries.add((seat, 'back_jump'))
            if new['action'] in (201, 203):
                movement_entries.add((seat, 'back_d'))
        for seat, kind in movement_entries:
            row = self.totals[seat]
            if kind == 'back_jump':
                self.streak[seat] += 1
                row['back_jumps'] += 1
                row['max_back_jump_streak'] = max(row['max_back_jump_streak'], self.streak[seat])
                cost = min(c.back_jump_cap_hp, c.back_jump_hp * (self.streak[seat] - 1))
                parts[seat]['back_jump'] -= cost
                row['back_jump_cost'] += cost
            else:
                parts[seat]['back_d'] -= c.back_d_hp
                row['back_d'] += 1
                row['back_d_cost'] += c.back_d_hp
        return parts


class CommandExecution:
    """One selected command; cancellation releases control, never fakes success."""
    def __init__(self, program, state):
        self.player = CommandPlayer(program)
        self.is_attack = any(isinstance(s, Keys) and any(b in s.buttons for b in 'ABC') for s in program)
        self.start_action = state['action']
        self.accepted = False
        self.interrupted = False
        self.cancel_open = False

    def action(self, frame, state):
        return encode_action(self.player.next(frame, state['facing'], segment=state['segment'], state=state))

    def observe(self, events, seat, state):
        self.player.observe_native(events, seat)
        self.accepted |= any(e['kind'] == 'ACTION_TRANSITION' and e['attacker_seat'] == seat
                             and e.get('accepted') and e.get('native_input_scope')
                             and e.get('lineage') == 'DIRECT'
                             and (not self.is_attack or e['action_after'] >= 300) for e in events)
        self.interrupted |= 50 <= state['action'] < 200
        self.cancel_open = bool(state['frame_flags'] & 0x20) and state['hitstop'] == 0

    @property
    def done(self):
        # No forced recovery wait: the policy chooses the next single command.
        return self.player.done or self.interrupted

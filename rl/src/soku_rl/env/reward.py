"""Opt-in native-event reward; no inferred collision or damage calculation."""
from dataclasses import dataclass
from math import isfinite

from soku_rl.env.combat_events import counter_hit_bonus


@dataclass(frozen=True)
class CombatRewardConfig:
    combo_beta: float = .3
    counter_coefficient: float = .2
    passive_b: float = .02
    passive_exponent: float = 2.
    passive_grace: int = 120
    technical_cap: float = 5000.
    terminal_win: float = 10000.
    terminal_loss: float = -10000.
    terminal_timeout: float = -10000.
    scale: float = 10000.
    passive_cap: float = 2500.
    unknown_combo_policy: str = 'error'
    pursuit_distance: float = 160.
    pursuit_minimum_progress: float = .01

    def __post_init__(self):
        if not all(isfinite(value) for key,value in self.__dict__.items() if key!='unknown_combo_policy'):
            raise ValueError('reward coefficients must be finite')
        if (not 0 <= self.combo_beta <= .3 or not 0 <= self.counter_coefficient <= .2
                or self.passive_b < 0 or self.passive_exponent < 1
                or type(self.passive_grace) is not int or self.passive_grace <= 0
                or self.technical_cap < 0 or self.scale <= 0 or self.passive_cap<0
                or self.pursuit_distance<0 or self.pursuit_minimum_progress<0
                or self.unknown_combo_policy not in ('error','no_bonus')):
            raise ValueError('invalid native reward coefficients')


class CombatReward:
    """Per-instance chronological ledger, in HP units until final normalization."""
    def __init__(self, config):
        self.config = config
        self.frame = None

    def reset(self, time_step):
        self.frame, self.segment = time_step.frame, time_step.diagnostics['segment']
        self.hp = tuple(time_step.diagnostics['hp'])
        self.passive = [0, 0]
        self.technical = [0., 0.]
        self.passive_spent = [0.,0.]
        self.totals=[dict(hits=0,blocks=0,combo_extensions=0,later_segment_hits=0,counter_hits=0,
            unknown_lineage_hits=0,damage_dealt=0.,damage_taken=0.,passive_frames=0,
            passive_cost=0.,combo_bonus=0.,counter_bonus=0.,terminal=0.) for _ in range(2)]
        self.ended = False

    def step(self, time_step, *, truncated=False):
        if (self.ended or time_step.frame != self.frame+1
                or time_step.diagnostics['segment'] != self.segment):
            raise RuntimeError('reward requires consecutive frames in one episode')
        if 'native_combat_events' not in time_step.diagnostics:
            raise NotImplementedError('native combat sidecar is required for Reward V2')
        c = self.config
        parts = [dict(damage=0.,combo=0.,counter=0.,passive=0.,terminal=0.) for _ in range(2)]
        hp = list(self.hp)
        technical = self.technical.copy()
        totals=[row.copy() for row in self.totals]
        seen = set()
        previous_order = 0
        for event in time_step.diagnostics['native_combat_events']:
            if event['frame'] != time_step.frame or event['segment'] != self.segment:
                raise RuntimeError('reward event frame/epoch mismatch')
            key = event['order']
            if key in seen:
                raise RuntimeError('duplicate native event')
            seen.add(key)
            if key <= previous_order:
                raise RuntimeError('native events must be in contact order')
            previous_order = key
            if event['kind'] not in ('HIT_RESOLUTION','BLOCK_HIGH_RESOLUTION','BLOCK_LOW_RESOLUTION'):
                continue
            attacker, defender = event['attacker_seat'], event['defender_seat']
            if attacker not in (0,1) or defender not in (0,1) or attacker == defender:
                raise RuntimeError('invalid native contact ownership')
            if event['hp_before'] != hp[defender] or event['hp_after'] > hp[defender]:
                raise RuntimeError('uncovered HP mutation or unordered contacts')
            damage = hp[defender]-event['hp_after']
            hp[defender] = event['hp_after']
            parts[attacker]['damage'] += damage
            parts[defender]['damage'] -= damage
            totals[attacker]['damage_dealt']+=damage
            totals[defender]['damage_taken']+=damage
            totals[attacker]['blocks' if event['hit_or_block']=='BLOCK' else 'hits']+=int(damage>0 or event['hit_or_block']=='BLOCK')
            if not damage or event['hit_or_block'] != 'HIT':
                continue
            combo = counter = 0.
            if c.combo_beta:
                if event.get('input_lineage_evidence')=='DIAGNOSTIC_ACTION_MAPPING':
                    raise NotImplementedError('diagnostic action mapping cannot authorize combo reward')
                segment = event.get('independent_input_segment')
                rate = event.get('damage_rate')
                if type(segment) is not bool:
                    totals[attacker]['unknown_lineage_hits']+=1
                    if c.unknown_combo_policy=='error':
                        raise NotImplementedError('independent second-or-later attack segment is unproven')
                    # Explicit conservative policy: unknown is logged and never
                    # earns a technical bonus; it is not relabeled as proven.
                    segment=False
                if rate is None:raise NotImplementedError('native damage-time Rate unavailable')
                if not isfinite(rate) or not 0 <= rate <= 1:
                    raise ValueError('native Rate outside supported reward domain')
                combo = damage*c.combo_beta*segment*(1-rate)
                totals[attacker]['later_segment_hits']+=int(segment)
                totals[attacker]['combo_extensions']+=int(segment and event.get('new_independent_attack',False))
            if c.counter_coefficient:
                counter = counter_hit_bonus(event,c.counter_coefficient)
            remaining = max(0.,c.technical_cap-technical[attacker])
            fraction = min(1.,remaining/(combo+counter)) if combo+counter else 1.
            parts[attacker]['combo'] += combo*fraction
            parts[attacker]['counter'] += counter*fraction
            technical[attacker] += (combo+counter)*fraction
            totals[attacker]['counter_hits']+=int(counter>0)
            totals[attacker]['combo_bonus']+=combo*fraction
            totals[attacker]['counter_bonus']+=counter*fraction
        if tuple(hp) != tuple(time_step.diagnostics['hp']):
            raise RuntimeError('native contacts do not account for all HP changes')
        passive = self.passive.copy()
        spent=self.passive_spent.copy()
        if c.passive_b:
            qualified = time_step.diagnostics.get('qualified_passive')
            if (not isinstance(qualified,(tuple,list)) or len(qualified)!=2
                    or any(type(value) is not bool for value in qualified)):
                raise NotImplementedError('forced-state/pursuit-qualified passive evidence unavailable')
            def cost(t):
                ratio=max(0,t-c.passive_grace)/c.passive_grace
                threshold=(c.passive_cap/(10000*c.passive_b))**(1/c.passive_exponent)
                return c.passive_cap if ratio>=threshold else 10000*c.passive_b*ratio**c.passive_exponent
            for seat,value in enumerate(qualified):
                reset=time_step.diagnostics.get('passive_reset')
                if reset is not None and (not isinstance(reset,(list,tuple)) or len(reset)!=2
                        or any(type(flag) is not bool for flag in reset)):
                    raise ValueError('invalid native passive reset evidence')
                if reset is not None and reset[seat]:
                    passive[seat]=0
                elif value:
                    increment=min(max(0.,c.passive_cap-spent[seat]),cost(passive[seat]+1)-cost(passive[seat]))
                    parts[seat]['passive'] = -increment
                    spent[seat]+=increment
                    passive[seat] += 1
                    totals[seat]['passive_frames']+=1
                    totals[seat]['passive_cost']+=increment
                elif reset is None:
                    passive[seat] = 0
        if time_step.terminated:
            if time_step.outcome.value in ('p1_win','p2_win'):
                winner = 0 if time_step.outcome.value=='p1_win' else 1
                parts[winner]['terminal']=c.terminal_win
                parts[1-winner]['terminal']=c.terminal_loss
        elif truncated or time_step.truncated:
            for part in parts: part['terminal']=c.terminal_timeout
        self.frame, self.hp = time_step.frame, tuple(hp)
        self.passive, self.technical = passive, technical
        self.passive_spent=spent
        for seat in (0,1):totals[seat]['terminal']+=parts[seat]['terminal']
        self.totals=totals
        self.ended = time_step.ended or truncated
        return tuple(sum(part.values())/c.scale for part in parts), tuple(parts)

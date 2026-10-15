"""Native snapshot evidence, deliberately not a guessed contact classifier."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CombatEvent:
    frame: int
    seat: int
    kind: str
    before: object
    after: object
    evidence: str = 'NATIVE_FRAME_BOUNDARY'
    attacker_instance: str | None = None
    contact_order: int | None = None
    hit_or_block: str = 'UNKNOWN'
    damage_rate: float | None = None
    punish_counter: bool | None = None


def player_projection(player):
    return dict(hp=int(player.hp), action=(int(player.actionId), int(player.sequenceId)),
                combo=(int(player.comboCount), float(player.comboRate),
                       int(player.comboDamage), int(player.comboLimit)))


class CombatEventTracker:
    """Each instance owns one chronology; resets emit no synthetic damage."""
    def __init__(self):
        self.previous = None
        self.segment = None
        self.frame = None

    def observe(self, raw):
        frame, segment = int(raw.frameId), int(raw.segmentId)
        current = tuple(player_projection(p) for p in (raw.p1, raw.p2))
        if frame == 0:
            self.previous, self.segment, self.frame = current, segment, frame
            return ()
        if self.previous is None or segment != self.segment or frame != self.frame+1:
            raise RuntimeError('combat events require reset then consecutive native frames')
        events = []
        for seat, (before, after) in enumerate(zip(self.previous, current)):
            for key, kind in (('hp', 'HP_CHANGE'), ('action', 'ACTION_STATE_CHANGE'),
                              ('combo', 'COMBO_STATE_CHANGE')):
                if before[key] != after[key]:
                    events.append(CombatEvent(frame, seat, kind, before[key], after[key]))
        self.previous, self.frame = current, frame
        return tuple(asdict(event) for event in events)


def require_resolved_contact(event):
    """Reward V2 must fail closed until native attribution and Rate are proven."""
    if (event['hit_or_block'] not in {'HIT', 'BLOCK'} or event['attacker_instance'] is None
            or event['contact_order'] is None or event['damage_rate'] is None):
        raise NotImplementedError('contact attribution/pre-damage Rate are not available')


def counter_hit_bonus(event, coefficient=0.2):
    """User-defined Punish proxy: only first-hit native CH, in damage units."""
    if not 0 <= coefficient <= 1:
        raise ValueError('counter bonus coefficient must be within [0, 1]')
    if event.get('hit_or_block') != 'HIT':
        return 0.0
    flag = event.get('counter_hit_first')
    if flag is None:
        raise NotImplementedError('first-hit native Counter Hit is unavailable')
    return coefficient * max(0, event['hp_before']-event['hp_after']) if flag else 0.0

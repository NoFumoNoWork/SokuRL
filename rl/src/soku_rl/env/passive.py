"""Reward-only eligibility from shared native state, never policy observations."""


class PassiveQualifier:
    def __init__(self, pursuit_distance=160., minimum_progress=.01):
        if pursuit_distance<0 or minimum_progress<0:
            raise ValueError('nonnegative reward-policy thresholds required')
        self.distance,self.progress=pursuit_distance,minimum_progress
        self.previous=None

    def observe(self,raw,events):
        positions=(float(raw.p1.x),float(raw.p2.x))
        if int(raw.frameId)==0:
            self.previous=positions
            return dict(qualified_passive=(False,False),passive_reset=(True,True),
                        passive_reason=('reset','reset'))
        if self.previous is None:raise RuntimeError('reset passive qualification first')
        interacted=any(e['kind'] in ('HIT_RESOLUTION','BLOCK_HIGH_RESOLUTION','BLOCK_LOW_RESOLUTION')
                       and e.get('hit_or_block') in ('HIT','BLOCK') for e in events)
        eligible=[];reasons=[]
        for seat,player in enumerate((raw.p1,raw.p2)):
            # SokuLib Action.hpp common hit/block/knockdown/tech range, not move IDs.
            forced=(50<=int(player.actionId)<200 or bool(int(player.frameFlags)&8)
                    or player.hitstop>0 or player.untech>0 or player.timeStop>0)
            toward=(positions[seat]-self.previous[seat])*(1 if positions[1-seat]>positions[seat] else -1)
            pursuit=(abs(positions[0]-positions[1])>self.distance and toward>self.progress)
            reason='interaction' if interacted else 'forced' if forced else 'pursuit' if pursuit else 'no_contact'
            eligible.append(reason=='no_contact');reasons.append(reason)
        self.previous=positions
        return dict(qualified_passive=tuple(eligible),passive_reset=(interacted,interacted),
                    passive_reason=tuple(reasons),
                    passive_evidence='SHARED_NATIVE_FORCED_STATE_AND_ACTUAL_POSITION_PROGRESS')

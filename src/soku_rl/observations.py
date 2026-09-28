"""Convert exported game frames to immutable, player-relative observations."""
from .baselines import Fighter, Observation, Projectile


def observe(state, player_index):
    if player_index not in (0, 1):
        raise ValueError("player_index must be zero or one")
    if state.p1ObjectOverflow or state.p2ObjectOverflow:
        raise RuntimeError("object observation overflow; this episode is invalid")
    players = (state.p1, state.p2)

    def fighter(raw):
        if raw.maxSpirit == 0:
            raise RuntimeError("player maxSpirit is zero")
        return Fighter(raw.x, raw.y, raw.hp, raw.spirit / raw.maxSpirit,
                       raw.actionId, bool(raw.airborne), raw.hitstop, raw.characterId, raw.facing)

    objects, count = ((state.p2Objects, state.p2ObjectCount) if player_index == 0
                      else (state.p1Objects, state.p1ObjectCount))
    projectiles = tuple(Projectile(obj.x, obj.y, obj.speedX, obj.speedY)
                        for obj in objects[:count] if obj.isActive and obj.hitBoxCount)
    return Observation(state.frameId, fighter(players[player_index]),
                       fighter(players[1 - player_index]), projectiles)

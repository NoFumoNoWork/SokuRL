"""Convert exported game frames to immutable, player-relative observations."""
from dataclasses import dataclass
from soku_rl.env.observation.gauges import spirit_fraction


@dataclass(frozen=True, slots=True)
class Fighter:
    x: float
    y: float
    hp: int
    spirit_fraction: float
    action_id: int
    airborne: bool
    hitstop: int
    character_id: int
    facing: int


@dataclass(frozen=True, slots=True)
class Projectile:
    x: float
    y: float
    speed_x: float
    speed_y: float


@dataclass(frozen=True, slots=True)
class Observation:
    frame: int
    player: Fighter
    opponent: Fighter
    enemy_projectiles: tuple[Projectile, ...]

def observe(state, player_index):
    if player_index not in (0, 1):
        raise ValueError("player_index must be zero or one")
    if state.p1ObjectOverflow or state.p2ObjectOverflow:
        raise RuntimeError("object observation overflow; this episode is invalid")
    objects, count = ((state.p2Objects, state.p2ObjectCount) if player_index == 0
                      else (state.p1Objects, state.p1ObjectCount))
    projectiles = tuple(Projectile(obj.x, obj.y, obj.speedX, obj.speedY)
                        for obj in objects[:count] if obj.isActive and obj.hitBoxCount)
    return observe_projectiles(state, player_index, projectiles)


def observe_projectiles(state, player_index, projectiles):
    """Combine bridge fighters with a complete, ordered enemy projectile list."""
    if player_index not in (0, 1):
        raise ValueError("player_index must be zero or one")
    players = (state.p1, state.p2)

    def fighter(raw):
        return Fighter(raw.x, raw.y, raw.hp, spirit_fraction(raw.spirit, raw.maxSpirit),
                       raw.actionId, bool(raw.airborne), raw.hitstop, raw.characterId, raw.facing)

    return Observation(state.frameId, fighter(players[player_index]),
                       fighter(players[1 - player_index]), projectiles)

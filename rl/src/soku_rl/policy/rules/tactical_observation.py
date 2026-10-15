"""Convert each supported observation into the inputs used by tactical rules."""
from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np

from soku_rl.policy.rules.baselines import TreePolicy
from soku_rl.env.observation.visible_state import STATE_FEATURES


@dataclass(frozen=True, slots=True)
class TacticalView:
    frame: int
    stride: int
    x: float
    y: float
    enemy_x: float
    enemy_y: float
    hp: float
    enemy_hp: float
    spirit: float
    enemy_spirit: float
    facing: int
    character: int
    visible: bool
    threatened: bool
    interrupted: bool
    arena_min: float
    arena_max: float

    @property
    def toward(self):
        return 1 if self.enemy_x >= self.x else -1

    @property
    def distance(self):
        return abs(self.enemy_x - self.x)

    @property
    def height(self):
        return self.enemy_y - self.y


def diagnostic_view(observation, movement):
    own, enemy = observation.player, observation.opponent
    if type(observation.frame) is not int or observation.frame < 0:
        raise ValueError("tactical frame must be a nonnegative integer")
    if own.character_id not in (0, 1):
        raise ValueError("tactical rules can control only Reimu and Marisa")
    for fighter in (own, enemy):
        if not all(isfinite(v) for v in (fighter.x, fighter.y, fighter.hp,
                                        fighter.spirit_fraction)):
            raise ValueError("invalid tactical fighter values")
        if fighter.character_id not in range(20) or fighter.facing not in (-1, 1):
            raise ValueError("tactical rules require a playable character with a valid facing")
        if not 0 <= fighter.spirit_fraction <= 1:
            raise ValueError("invalid tactical spirit fraction")
    for projectile in observation.enemy_projectiles:
        if not all(isfinite(v) for v in (projectile.x, projectile.y,
                                        projectile.speed_x, projectile.speed_y)):
            raise ValueError("invalid tactical projectile")
    return TacticalView(observation.frame, 1, own.x, own.y, enemy.x, enemy.y,
        own.hp / 10000, enemy.hp / 10000, own.spirit_fraction, enemy.spirit_fraction,
        own.facing, own.character_id, True,
        TreePolicy(movement)._projectile_threat(observation),
        50 <= own.action_id < 200, movement.arena_min, movement.arena_max)


def screen_view(values, frame, stride, movement, scale):
    if (values.shape != (STATE_FEATURES,) or not np.isfinite(values).all()
            or (abs(values) > 1).any()):
        raise ValueError("invalid public tactical frame")
    own, enemy = values[:8], values[8:16]
    if not any(np.isclose(own[7], character / 19) for character in (0, 1)):
        raise ValueError("tactical rules can control only Reimu and Marisa")
    for fighter in (own, enemy):
        if fighter[0] not in (0, 1) or not 0 <= fighter[5] <= 1 or not 0 <= fighter[6] <= 1:
            raise ValueError("invalid public tactical visibility or gauges")
        if fighter[0] == 1 and fighter[3] not in (-1, 1):
            raise ValueError("visible tactical fighter requires a valid facing")
        character = round(float(fighter[7]) * 19)
        if character not in range(20) or not np.isclose(fighter[7], character / 19):
            raise ValueError("tactical rules require a valid playable character ID")
    # Screen Y points down. Scaling does not recover the hidden world camera.
    x, y = float(own[1]) * 640 / scale, -float(own[2]) * 480 / scale
    enemy_x, enemy_y = float(enemy[1]) * 640 / scale, -float(enemy[2]) * 480 / scale
    objects = values[16 + 64 * 3:].reshape(-1, 3)
    threatened = any(p[0] == 1 and hypot(float(p[1]) * 640 / scale - x,
        -float(p[2]) * 480 / scale - (y + 25 / scale)) <= movement.projectile_radius
        for p in objects)
    return TacticalView(frame, stride, x, y, enemy_x, enemy_y,
        float(own[5]), float(enemy[5]), float(own[6]), float(enemy[6]),
        int(own[3]), round(float(own[7]) * 19), own[0] == 1 and enemy[0] == 1,
        threatened, False, 0., 640 / scale)

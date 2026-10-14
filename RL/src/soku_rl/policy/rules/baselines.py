"""Deterministic decision trees over a player-relative battle observation."""
from dataclasses import dataclass
from math import hypot, isfinite

from soku_rl.env.encoding import Decision
from soku_rl.env.observation.diagnostic import Observation


@dataclass(frozen=True, slots=True)
class TreeConfig:
    style: str
    melee_range: float
    desired_min: float
    desired_max: float
    vertical_tolerance: float
    spirit_reserve: float
    attack_interval: int
    projectile_horizon: float
    projectile_radius: float
    arena_min: float
    arena_max: float
    wall_margin: float

    def __post_init__(self):
        if self.style not in {"idle", "rush", "zoning", "counter"}:
            raise ValueError(f"unknown tree style: {self.style}")
        if not 0 < self.melee_range <= self.desired_min < self.desired_max:
            raise ValueError("expected 0 < melee_range <= desired_min < desired_max")
        if not 0 <= self.spirit_reserve <= 1 or self.attack_interval < 2:
            raise ValueError("invalid spirit reserve or attack interval")
        if not self.arena_min < self.arena_max or self.wall_margin <= 0:
            raise ValueError("invalid arena bounds")
        if min(self.vertical_tolerance, self.projectile_horizon, self.projectile_radius) <= 0:
            raise ValueError("vertical tolerance and projectile settings must be positive")


def _decision(horizontal, vertical, button, rule):
    buttons = [0, 0, 0, 0, 0, 0]
    if button:
        buttons[{"A": 0, "B": 1, "C": 2, "D": 3}[button]] = 1
    return Decision((horizontal, vertical, *buttons), rule)


class TreePolicy:
    """One call per simulated frame; button pulses use the observation frame."""

    def __init__(self, config: TreeConfig):
        self.config = config

    def act(self, observation: Observation) -> Decision:
        own, enemy = observation.player, observation.opponent
        if observation.frame < 0 or not all(isfinite(v) for v in (
                own.x, own.y, own.spirit_fraction, enemy.x, enemy.y)):
            raise ValueError("observation contains an invalid frame or numeric value")
        cfg = self.config
        if cfg.style == "idle":
            return _decision(0, 0, "", "idle")
        if own.hp <= 0 or enemy.hp <= 0 or own.hitstop:
            return _decision(0, 0, "", "terminal_or_hitstop")
        toward = 1 if enemy.x >= own.x else -1
        distance = abs(enemy.x - own.x)
        retreat_blocked = (own.x <= cfg.arena_min + cfg.wall_margin and toward > 0
                           or own.x >= cfg.arena_max - cfg.wall_margin and toward < 0)
        threat = self._projectile_threat(observation)
        if threat and own.spirit_fraction > cfg.spirit_reserve:
            direction = toward if cfg.style == "rush" or retreat_blocked else -toward
            return _decision(direction, -1, "D", "graze_projectile")
        if own.spirit_fraction < cfg.spirit_reserve and distance > cfg.melee_range:
            return _decision(toward if retreat_blocked else -toward, 0, "", "recover_spirit")
        if cfg.style == "counter" and 300 <= enemy.action_id < 400 and distance < cfg.desired_min:
            # Action numbers come from the pinned SokuLib Action.hpp. This
            # observes a melee animation; it does not predict active hitboxes.
            return _decision(-toward, 0 if enemy.airborne else 1, "", "guard_melee")
        aligned = abs(enemy.y - own.y) <= cfg.vertical_tolerance
        if distance <= cfg.melee_range and aligned:
            return self._attack(observation.frame, "A", "close_melee")
        if enemy.y > own.y + cfg.vertical_tolerance and not own.airborne:
            return _decision(toward, -1, "", "follow_airborne_opponent")
        if cfg.style == "rush":
            return _decision(toward, 0, "D" if distance > cfg.desired_max else "", "approach")
        if cfg.style == "zoning" and distance < cfg.desired_min:
            if retreat_blocked:
                return _decision(toward, -1, "D", "escape_corner")
            return _decision(-toward, 0, "", "make_space")
        if distance > cfg.desired_max:
            dash = cfg.style == "counter" and enemy.action_id >= 400
            return _decision(toward, 0, "D" if dash else "", "close_distance")
        if cfg.style == "counter" and enemy.action_id >= 400 and distance > cfg.desired_min:
            return _decision(toward, 0, "D", "approach_during_shot")
        button = "C" if cfg.style == "zoning" and observation.frame // cfg.attack_interval % 3 == 2 else "B"
        return self._attack(observation.frame, button, "ranged_attack")

    def _attack(self, frame, button, rule):
        if frame % self.config.attack_interval:
            return _decision(0, 0, "", "release_attack")
        return _decision(0, 0, button, rule)

    def _projectile_threat(self, observation):
        for projectile in observation.enemy_projectiles:
            dx = projectile.x - observation.player.x
            dy = projectile.y - (observation.player.y + 50)
            speed_squared = projectile.speed_x ** 2 + projectile.speed_y ** 2
            closest_time = 0.0
            if speed_squared:
                closest_time = max(0.0, min(self.config.projectile_horizon,
                    -(dx * projectile.speed_x + dy * projectile.speed_y) / speed_squared))
            if hypot(dx + projectile.speed_x * closest_time,
                     dy + projectile.speed_y * closest_time) <= self.config.projectile_radius:
                return True
        return False

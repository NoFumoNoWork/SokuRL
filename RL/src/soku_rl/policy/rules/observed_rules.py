"""Run rule policies from the same observation tensors given to learners."""
from collections import deque
from dataclasses import asdict, dataclass, replace
from functools import cached_property
import hashlib
import json
import math

import numpy as np

from soku_rl.env.observation.diagnostic import Fighter, Observation, Projectile
from soku_rl.policy.rules.baselines import _decision
from soku_rl.env.wrappers.learning import LearningInterface
from soku_rl.env.encoding import FRAME_FEATURES, FIGHTER_SCALES, encode_action
from soku_rl.policy.rules.strategies import strategy_from_config
from soku_rl.policy.rules.tactical_observation import screen_view
from soku_rl.env.observation.visible_state import STATE_FEATURES
from soku_rl.policy.base import RulePolicy as RulePolicyBase


def decode_diagnostic(values, horizon):
    """Invert the documented diagnostic encoding, without another memory read."""
    if values.shape != (FRAME_FEATURES,) or not np.isfinite(values).all():
        raise ValueError("invalid diagnostic frame")
    fighters = []
    for index in range(2):
        raw = values[1 + index * 9:10 + index * 9] * FIGHTER_SCALES
        fighters.append(Fighter(float(raw[0]), float(raw[1]), round(float(raw[2])),
            float(raw[3]), round(float(raw[4])), bool(round(float(raw[5]))),
            round(float(raw[6])), round(float(raw[7])), round(float(raw[8]))))
    projectiles = tuple(Projectile(float(p[1] * 1280), float(p[2] * 1280),
                                   float(p[3] * 100), float(p[4] * 100))
                        for p in values[19:].reshape(-1, 5) if p[0] == 1)
    return Observation(round(float(values[0]) * horizon), *fighters, projectiles)


@dataclass(frozen=True)
class RulePolicy(RulePolicyBase):
    name: str
    rules: dict
    episode: object
    implementation: str

    def __post_init__(self):
        if self.name == "god":
            self.god_policy
            return
        if self.episode.observation_mode not in {"state", "diagnostic_state", "privileged_state"}:
            raise ValueError("rule policies require a numeric observation")
        if self.episode.observation_mode == "diagnostic_state" and self.episode.decision_frames != 1:
            raise ValueError("legacy diagnostic rules require one decision per frame")
        strategy_from_config(self.name, self.rules, self.implementation)

    @property
    def fingerprint(self):
        if self.name == "god":
            return self.god_policy.fingerprint
        data = [self.name, self.rules, asdict(self.episode), self.implementation]
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def spawn(self, seed):
        if self.name == "god":
            return self.god_policy.spawn(seed)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("policy seed must be a uint32")
        strategy = strategy_from_config(self.name, self.rules, self.implementation)
        actor = strategy.spawn(seed)
        if self.episode.observation_mode == "state" and strategy.kind == "tactical":
            actor = ScreenTactics(actor, self.rules["screen"], self.episode.decision_frames)
        elif self.episode.observation_mode == "state":
            actor = ScreenRules(strategy, self.rules["screen"], self.episode.decision_frames)
        return RuleEpisode(actor, self.episode)

    @cached_property
    def god_policy(self):
        from soku_rl.policy.god.runtime import GodPolicy
        from soku_rl.policy.god.package import ScriptPackage
        config = self.rules["god"]
        return GodPolicy(self.name, ScriptPackage(config["package"], config["api_source"]),
                         config["script"], self.episode)

    def spawn_play(self, seed):
        if self.name == "god":
            return self.god_policy.spawn_play(seed)
        return super().spawn_play(seed)


@dataclass
class RuleEpisode:
    actor: object
    episode: object

    def act(self, observation):
        values = np.asarray(observation)
        if self.episode.observation_mode == "privileged_state":
            from soku_rl.env.observation.privileged import decode_privileged
            from soku_rl.env.observation.memory_schema import PRIVILEGED_FEATURES
            current = decode_privileged(values[-PRIVILEGED_FEATURES:])
            fighters = [Fighter(p["x"], p["y"], int(p["hp"]), p["rei"] / 1000.,
                int(p["act"]), bool(int(p["fflags"]) & 4), int(p["hitstop"]), int(p["char"]), int(p["dir"]))
                for p in current.players]
            objects = tuple(Projectile(p["x"], p["y"], p["xspeed"], p["yspeed"])
                            for p in current.players[1]["objects"] if p["attackarea_n"])
            return encode_action(self.actor.act(Observation(int(current.world["frame"]), *fighters, objects)).inputs)
        width = STATE_FEATURES if self.episode.observation_mode == "state" else FRAME_FEATURES
        if values.shape != (width * self.episode.history_frames,) or not np.isfinite(values).all():
            raise ValueError("rule observation does not match the episode configuration")
        last = values[-width:]
        current = last if self.episode.observation_mode == "state" else decode_diagnostic(
            last, self.episode.max_frames)
        return encode_action(self.actor.act(current).inputs)


class ScreenTactics:
    """Advance tactical rules once per public observation, at learner cadence."""

    def __init__(self, actor, screen, decision_frames):
        if (not math.isfinite(screen["distance_scale"]) or screen["distance_scale"] <= 0
                or type(decision_frames) is not int or decision_frames < 1):
            raise ValueError("invalid tactical screen scale or decision duration")
        self.actor, self.scale, self.stride = actor, screen["distance_scale"], decision_frames
        self.frame = -decision_frames

    def act(self, values):
        self.frame += self.stride
        return self.actor.act_view(screen_view(values, self.frame, self.stride,
                                              self.actor.movement, self.scale))


class ScreenRules:
    """Screen-only variants: no action IDs, hidden poses or hitstop counters.

    Each queued command consumes one decision, so latency and held-key duration
    are exactly those of the learner. Screen Y increases downwards.
    """
    def __init__(self, strategy, screen, decision_frames):
        values = json.loads(strategy.config_json)
        self.style = strategy.name
        self.movement = values["movement"] if strategy.kind == "community" else values
        self.special = values["rules"] if strategy.kind == "community" else {}
        if set(screen) != {"distance_scale", "damage_guard_frames"}:
            raise ValueError("screen rule configuration has unexpected fields")
        if screen["distance_scale"] <= 0 or screen["damage_guard_frames"] < 0:
            raise ValueError("invalid screen rule distances or guard duration")
        self.scale, self.guard_frames = screen["distance_scale"], screen["damage_guard_frames"]
        self.stride = decision_frames
        self.frame = -decision_frames
        self.next_attack = self.next_special = self.guard_until = 0
        self.previous_hp = 1.
        self.pending = deque()

    def act(self, values):
        if values.shape != (STATE_FEATURES,) or (abs(values) > 1).any():
            raise ValueError("invalid public screen frame")
        self.frame += self.stride
        own, enemy = values[:8], values[8:16]
        damaged = own[5] < self.previous_hp
        self.previous_hp = float(own[5])
        if damaged:
            self.pending.clear()
            self.guard_until = self.frame + self.guard_frames
        if self.style == "idle" or own[5] <= 0 or enemy[5] <= 0:
            return _decision(0, 0, "", "terminal_or_idle")
        if own[0] != 1 or enemy[0] != 1:
            self.pending.clear()
            return _decision(0, 0, "", "pose_not_visible")
        x, y, ex, ey = float(own[1] * 640), float(own[2] * 480), float(enemy[1] * 640), float(enemy[2] * 480)
        toward = 1 if ex >= x else -1
        distance = abs(ex - x)
        cfg = self.movement
        near, low, high, vertical, radius = (cfg[k] * self.scale for k in (
            "melee_range", "desired_min", "desired_max", "vertical_tolerance", "projectile_radius"))
        wall = cfg["wall_margin"] * self.scale
        corner = x <= wall and toward > 0 or x >= 640 - wall and toward < 0
        enemy_objects = values[16 + 64 * 3:].reshape(-1, 3)
        threatened = any(p[0] == 1 and math.hypot(p[1] * 640 - x, p[2] * 480 - (y - 25)) <= radius
                         for p in enemy_objects)
        if threatened and own[6] > cfg["spirit_reserve"]:
            self.pending.clear()
            return _decision(toward if cfg["style"] == "rush" or corner else -toward,
                             -1, "D", "graze_visible_object")
        if self.style in {"counter", "community_guard"} and self.frame < self.guard_until and distance < low:
            return _decision(-toward, int(abs(ey - y) <= vertical), "", "guard_after_visible_damage")
        if self.pending:
            return self.pending.popleft()
        aligned = abs(ey - y) <= vertical
        if self.special and aligned and own[3] == toward and own[6] >= .4 and self.frame >= self.next_special:
            if self.special["special_min"] * self.scale <= distance <= self.special["special_max"] * self.scale:
                self.next_special = self.frame + self.special["special_cooldown"]
                self.pending.extend((_decision(toward, 1, "", "236_down_forward"),
                                     _decision(toward, 0, "B", "236_forward_B"),
                                     _decision(0, 0, "", "236_release")))
                return _decision(0, 1, "", "236_down")
        if distance <= near and aligned:
            return self._attack("A")
        if own[6] < cfg["spirit_reserve"] and distance > near:
            return _decision(toward if corner else -toward, 0, "", "recover_spirit")
        if ey < y - vertical:
            return _decision(toward, -1, "", "follow_visible_opponent")
        if cfg["style"] == "rush":
            return _decision(toward, 0, "D" if distance > high else "", "approach")
        if cfg["style"] == "zoning" and distance < low:
            return _decision(toward if corner else -toward, -1 if corner else 0,
                             "D" if corner else "", "make_space")
        if distance > high:
            return _decision(toward, 0, "", "close_distance")
        return self._attack("C" if cfg["style"] == "zoning" else "B")

    def _attack(self, button):
        if self.frame < self.next_attack:
            return _decision(0, 0, "", "release_attack")
        self.next_attack = self.frame + max(self.movement["attack_interval"], 2 * self.stride)
        return _decision(0, 0, button, "visible_range_attack")


@dataclass(frozen=True)
class LearningRulePolicy(RulePolicyBase):
    """Map an existing rule policy into the same learner action vocabulary."""
    policy: object
    interface: LearningInterface

    @property
    def name(self):
        return self.policy.name

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps([self.policy.fingerprint,
            asdict(self.interface.config)], sort_keys=True).encode()).hexdigest()

    def spawn(self, seed):
        return LearningRuleEpisode(self.policy.spawn(seed), self.interface)

    def spawn_play(self, seed):
        instance = self.policy.spawn_play(seed)
        return replace(instance, actor=LearningRuleEpisode(instance.actor, self.interface))


@dataclass
class LearningRuleEpisode:
    actor: object
    interface: LearningInterface

    def act(self, observation):
        return self.interface.action(self.actor.act(self.interface.base_observation(observation)))

    def act_observation(self, observation):
        return self.interface.action(self.actor.act_observation(observation))

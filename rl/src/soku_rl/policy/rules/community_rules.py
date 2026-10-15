"""Stateful rule subsets studied in the community th123_ai character scripts."""
from collections import deque
from dataclasses import dataclass

from soku_rl.env.encoding import Decision
from soku_rl.env.observation.diagnostic import Observation
from soku_rl.policy.rules.baselines import TreeConfig, TreePolicy, _decision


@dataclass(frozen=True, slots=True)
class CommunityConfig:
    style: str
    guard_range: float
    special_min: float
    special_max: float
    special_cooldown: int
    wakeup_frames: int

    def __post_init__(self):
        if self.style not in {"community_combo", "community_guard"}:
            raise ValueError(f"unknown community style: {self.style}")
        if not 0 < self.special_min < self.special_max or self.guard_range <= 0:
            raise ValueError("invalid rule distances")
        if self.special_cooldown < 8 or self.wakeup_frames < 1:
            raise ValueError("invalid rule durations")


class CommunityPolicy:
    """Independent partial reimplementation; this is not the original God AI.

    Each call advances one simulation frame. Create a new object per episode.
    Only Reimu and Marisa with their default skills are supported.
    """

    def __init__(self, config: CommunityConfig, movement: TreeConfig):
        self.config = config
        self.movement = TreePolicy(movement)
        self.pending = deque()
        self.last_frame = -1
        self.next_special = 0
        self.last_buttons = (0, 0, 0, 0, 0, 0)
        self.was_knocked_down = False

    def _emit(self, decision):
        self.last_buttons = decision.inputs[2:]
        return decision

    def _pulse(self, horizontal, button, rule):
        index = {"A": 0, "B": 1}[button]
        if self.last_buttons[index]:
            return self._emit(_decision(horizontal, 0, "", "release_for_" + rule))
        return self._emit(_decision(horizontal, 0, button, rule))

    def act(self, observation: Observation) -> Decision:
        own, enemy = observation.player, observation.opponent
        if own.character_id not in (0, 1) or enemy.character_id not in (0, 1):
            raise ValueError("community rule subsets support only Reimu and Marisa")
        if own.facing not in (-1, 1):
            raise ValueError("facing must be -1 or 1")
        if observation.frame != self.last_frame + 1:
            raise ValueError("community policy requires consecutive frames from zero")
        self.last_frame = observation.frame
        toward = 1 if enemy.x >= own.x else -1
        distance = abs(enemy.x - own.x)
        cfg = self.config
        if own.hp <= 0 or enemy.hp <= 0:
            self.pending.clear()
            return self._emit(_decision(0, 0, "", "terminal"))

        # A hit interrupts a stored command. Never replay its tail after recovery.
        knocked_down = own.action_id in (97, 98)
        if 50 <= own.action_id < 150:
            self.pending.clear()
            if knocked_down:
                direction = (1 if own.x < 500 else -1 if own.x > 780 else toward)
                if not self.was_knocked_down:
                    self.wakeup_until = observation.frame + cfg.wakeup_frames
                self.was_knocked_down = True
                if observation.frame < self.wakeup_until:
                    return self._emit(_decision(direction, 0, "D", "directional_wakeup"))
            return self._emit(_decision(0, 0, "", "wait_hitstun"))
        self.was_knocked_down = False

        # The Lua scripts confirm hits during the final 1..3 hitstop frames.
        # Input still has to be sent in hitstop, before animation can advance.
        confirmed = 50 <= enemy.action_id <= 89 or enemy.action_id == 143
        facing_enemy = own.facing == toward
        if 1 <= own.hitstop <= 3 and facing_enemy and confirmed:
            if own.action_id in (300, 330, 320, 321) and not enemy.airborne:
                self.pending.clear()
                return self._pulse(0, "A", "confirmed_A_chain")
            if own.action_id in (300, 330) and enemy.airborne and own.spirit_fraction >= 0.2:
                self.pending.clear()
                return self._pulse(toward, "B", "confirmed_air_6B")
        if self.pending:
            return self._emit(self.pending.popleft())

        aligned = abs(enemy.y - own.y) <= self.movement.config.vertical_tolerance
        if (cfg.style == "community_guard" and distance <= cfg.guard_range
                and aligned and 300 <= enemy.action_id < 400 and own.hitstop == 0):
            # Character-specific ground lows from 01_marisa_main.ai.
            # Reimu's delayed 3A is omitted because the needed timing is absent.
            low = enemy.action_id in ({303} if enemy.character_id == 0 else {303, 304})
            return self._emit(_decision(-toward, int(low and not own.airborne), "",
                                        "guard_low" if low else "guard_high"))
        if (not own.airborne and not enemy.airborne and own.action_id <= 5
                and facing_enemy and cfg.special_min <= distance <= cfg.special_max
                and own.spirit_fraction >= 0.4 and observation.frame >= self.next_special):
            self.next_special = observation.frame + cfg.special_cooldown
            # Same directional sequence as lvr236; captured facing stays fixed
            # until completion. Two button frames, then two release frames.
            self.pending.extend([
                _decision(0, 1, "", "236_down"),
                _decision(toward, 1, "", "236_down_forward"),
                _decision(toward, 0, "B", "236_forward_B"),
                _decision(toward, 0, "B", "236_hold_B"),
                _decision(0, 0, "", "236_release"),
                _decision(0, 0, "", "236_release"),
            ])
            return self._emit(self.pending.popleft())
        return self._emit(self.movement.act(observation))

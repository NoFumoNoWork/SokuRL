"""Ten stateful combat tactics with one decision engine for both state tracks."""
from collections import deque
from dataclasses import dataclass
from math import isfinite

from soku_rl.policy.rules.baselines import _decision
from soku_rl.policy.rules.tactical_observation import diagnostic_view


TACTICAL_STYLES = (
    "pressure", "footsies", "anti_air", "air_rush", "bullet_wall",
    "graze_hunter", "hit_and_run", "corner_trap", "spirit_siege", "skill_cycle",
)


@dataclass(frozen=True, slots=True)
class TacticalConfig:
    style: str
    guard_frames: int
    retreat_frames: int
    burst_frames: int
    special_cooldown: int
    spirit_recover: float
    shot_interval: int
    cycle_frames: int

    def __post_init__(self):
        if self.style not in TACTICAL_STYLES:
            raise ValueError(f"unknown tactical style: {self.style}")
        durations = (self.guard_frames, self.retreat_frames, self.burst_frames,
                     self.special_cooldown, self.shot_interval, self.cycle_frames)
        if any(type(v) is not int or v < 2 for v in durations):
            raise ValueError("tactical durations must be integers of at least two frames")
        if not isfinite(self.spirit_recover) or not 0 < self.spirit_recover <= 1:
            raise ValueError("invalid tactical spirit recovery threshold")


class TacticalPolicy:
    """Keep commands and tactical memory local to one episode.

    Tactics use relative geometry and changes in public gauges. The diagnostic
    adapter additionally detects input interruptions and projectile velocity.
    Character-specific inputs target Reimu and Marisa with default skills.
    """

    def __init__(self, config, movement):
        if config.spirit_recover <= movement.spirit_reserve:
            raise ValueError("spirit recovery threshold must exceed the reserve")
        self.config, self.movement = config, movement
        self.pending = deque()
        self.history = deque(maxlen=1)
        self.last_frame = -1
        self.next_attack = self.next_special = self.guard_until = 0
        self.retreat_until = self.burst_until = 0
        self.recovering = False
        self.attack_count = 0
        self.last_buttons = (0,) * 6
        self.choose = getattr(self, "_" + config.style)

    def act(self, observation):
        return self.act_view(diagnostic_view(observation, self.movement))

    def act_view(self, view):
        if (self.last_frame == -1 and view.frame != 0
                or self.last_frame != -1 and view.frame != self.last_frame + view.stride):
            raise ValueError("tactical policy requires consecutive decisions from zero")
        self.last_frame = view.frame
        decision = self._decide(view)
        self.last_buttons = decision.inputs[2:]
        return decision

    def _decide(self, s):
        if s.hp <= 0 or s.enemy_hp <= 0 or not s.visible:
            self.pending.clear()
            self.history.clear()
            self.burst_until = self.retreat_until = self.guard_until = 0
            return _decision(0, 0, "", "terminal_or_hidden")
        previous = self.history[0] if self.history else s
        damaged = s.hp < previous.hp
        confirmed = s.enemy_hp < previous.enemy_hp
        closing = s.distance < previous.distance
        crossed = s.toward != previous.toward
        self.history.append(s)
        if damaged or s.interrupted:
            self.pending.clear()
            self.guard_until = s.frame + self.config.guard_frames
        if crossed:
            self.pending.clear()
        if s.interrupted:
            return _decision(0, 0, "", "wait_recovery")
        if s.frame < self.guard_until and s.distance < self.movement.desired_min:
            return _decision(-s.toward, int(s.height <= self.movement.vertical_tolerance),
                             "", "guard_after_damage")
        if s.spirit < self.movement.spirit_reserve:
            self.recovering = True
            self.pending.clear()
        if s.spirit >= self.config.spirit_recover:
            self.recovering = False
        if self.recovering:
            if s.distance <= self.movement.melee_range:
                return self._attack(s, 0, 1, "A", "resource_defence")
            return self._retreat(s, "recover_spirit")
        if s.threatened:
            self.pending.clear()
            if self.config.style == "graze_hunter":
                self.burst_until = s.frame + self.config.burst_frames
                return _decision(s.toward, 0, "D", "hunt_through_bullets")
            toward = self.config.style in {"pressure", "air_rush", "corner_trap"}
            direction = s.toward if toward or self._corner(s) else -s.toward
            return _decision(direction, -1, "D", "evade_projectile")
        if self.pending:
            return self.pending.popleft()
        return self.choose(s, confirmed, closing)

    def _corner(self, s):
        return (s.x <= s.arena_min + self.movement.wall_margin and s.toward > 0
                or s.x >= s.arena_max - self.movement.wall_margin and s.toward < 0)

    def _retreat(self, s, rule):
        if self._corner(s):
            return _decision(s.toward, -1, "", "jump_out_of_corner")
        return _decision(-s.toward, 0, "", rule)

    def _attack(self, s, horizontal, vertical, button, rule):
        index = {"A": 0, "B": 1, "C": 2}[button]
        if s.frame < self.next_attack or self.last_buttons[index]:
            return _decision(0, 0, "", "release_attack")
        interval = self.movement.attack_interval if button == "A" else self.config.shot_interval
        self.next_attack = s.frame + max(interval, 2 * s.stride)
        self.attack_count += 1
        return _decision(horizontal, vertical, button, rule)

    def _pressure(self, s, confirmed, closing):
        if s.distance > self.movement.melee_range:
            return _decision(s.toward, -1 if s.height > self.movement.vertical_tolerance else 0,
                             "D" if s.distance > self.movement.desired_min else "", "pressure_chase")
        horizontal, vertical = ((0, 0) if confirmed else
                                ((0, 0), (0, 1), (s.toward, 0))[self.attack_count % 3])
        return self._attack(s, horizontal, vertical, "A", "pressure_chain" if confirmed else "pressure_mix")

    def _footsies(self, s, confirmed, closing):
        if s.distance < self.movement.desired_min and closing:
            self.burst_until = s.frame + self.config.burst_frames
            return self._retreat(s, "bait_approach")
        if s.frame < self.burst_until and not closing:
            if s.distance <= self.movement.desired_min:
                return self._attack(s, s.toward, 0, "A", "spacing_punish")
            return _decision(s.toward, 0, "D", "reenter_after_bait")
        if s.distance < self.movement.desired_min:
            return self._retreat(s, "keep_poke_distance")
        if s.distance > self.movement.desired_max:
            return _decision(s.toward, 0, "", "walk_into_poke_range")
        return self._attack(s, s.toward, 0, "A", "forward_poke")

    def _anti_air(self, s, confirmed, closing):
        if s.height > self.movement.vertical_tolerance:
            if s.distance <= self.movement.melee_range:
                return self._attack(s, -s.toward if s.character == 1 else 0, 0,
                                    "A", "close_anti_air")
            if s.distance <= self.movement.desired_max:
                return self._attack(s, 0, 1, "C", "anti_air_2C")
            return _decision(s.toward, 0, "", "track_below_opponent")
        if s.distance <= self.movement.melee_range:
            return self._attack(s, 0, 1, "A", "anti_air_ground_check")
        return self._attack(s, 0, 1, "B", "anti_air_2B_screen")

    def _air_rush(self, s, confirmed, closing):
        if s.height < -self.movement.vertical_tolerance:
            if s.distance <= self.movement.desired_min:
                return self._attack(s, 0, 1, "A", "descending_air_melee")
            return self._attack(s, 0, 1, "B", "descending_air_shot")
        if s.distance <= self.movement.melee_range:
            return self._attack(s, s.toward, 0, "A", "air_intercept")
        phase = s.frame % self.config.cycle_frames
        if phase < self.config.burst_frames:
            return _decision(s.toward, -1, "", "jump_approach")
        if phase < 2 * self.config.burst_frames:
            return _decision(s.toward, -1, "D", "air_flight_approach")
        return self._attack(s, s.toward, 0, "B", "air_cover_fire")

    def _bullet_wall(self, s, confirmed, closing):
        if s.distance < self.movement.desired_min:
            return self._retreat(s, "protect_firing_distance")
        if s.distance > self.movement.desired_max:
            return _decision(s.toward, 0, "", "enter_barrage_range")
        if s.frame < self.next_attack:
            return _decision(-s.toward, -1, "", "barrage_jump_cancel")
        phase = self.attack_count % 3
        return self._attack(s, 0, int(phase == 2), "B" if phase == 0 else "C", "layer_bullets")

    def _graze_hunter(self, s, confirmed, closing):
        if s.distance <= self.movement.desired_min and s.frame < self.burst_until:
            return self._attack(s, s.toward, 0, "A", "graze_dash_attack")
        if s.frame < self.burst_until:
            return _decision(s.toward, 0, "D", "continue_graze_chase")
        if s.distance <= self.movement.melee_range:
            return self._attack(s, 0, 0, "A", "hunter_melee")
        return self._attack(s, 0, 0, "B", "provoke_projectile_exchange")

    def _hit_and_run(self, s, confirmed, closing):
        if s.frame < self.retreat_until:
            return self._retreat(s, "withdraw_after_strike")
        if s.distance <= self.movement.desired_min:
            decision = self._attack(s, s.toward, 0, "A", "single_strike")
            if decision.inputs[2]:
                self.retreat_until = s.frame + self.config.retreat_frames
            return decision
        return _decision(s.toward, 0, "D", "raid_approach")

    def _corner_trap(self, s, confirmed, closing):
        pinned = (s.enemy_x <= s.arena_min + self.movement.wall_margin
                  or s.enemy_x >= s.arena_max - self.movement.wall_margin)
        if not pinned:
            if s.distance <= self.movement.desired_min:
                return self._attack(s, s.toward, 0, "A", "push_to_corner")
            return _decision(s.toward, 0, "D", "carry_to_corner")
        if s.height > self.movement.vertical_tolerance:
            return self._attack(s, 0, 1, "C", "deny_corner_jump")
        if s.distance > self.movement.desired_min:
            return _decision(s.toward, 0, "", "seal_corner_gap")
        phase = self.attack_count % 3
        return self._attack(s, 0, int(phase == 1), "C" if phase == 2 else "A", "corner_pressure")

    def _spirit_siege(self, s, confirmed, closing):
        if s.enemy_spirit <= self.movement.spirit_reserve:
            if s.distance <= self.movement.desired_max:
                return self._attack(s, s.toward, 0, "C", "exploit_low_spirit")
            return _decision(s.toward, 0, "D", "chase_depleted_spirit")
        if s.distance < self.movement.desired_min:
            return self._retreat(s, "siege_keep_distance")
        if s.distance > self.movement.desired_max:
            return _decision(s.toward, 0, "", "siege_advance")
        return self._attack(s, s.toward, 0, "B", "drain_guard_spirit")

    def _skill_cycle(self, s, confirmed, closing):
        if s.distance < self.movement.desired_min:
            if s.distance <= self.movement.melee_range:
                return self._attack(s, 0, 1, "A", "skill_defensive_low")
            return self._retreat(s, "skill_make_space")
        if s.distance > self.movement.desired_max:
            return _decision(s.toward, 0, "", "skill_enter_range")
        if (s.frame >= self.next_special and s.spirit >= .4 and s.facing == s.toward
                and abs(s.height) <= self.movement.vertical_tolerance):
            self.next_special = s.frame + self.config.special_cooldown
            button = "B" if self.attack_count % 2 == 0 else "C"
            self.attack_count += 1
            self.pending.extend((
                _decision(0, 1, "", "skill_236_down"),
                _decision(s.toward, 1, "", "skill_236_diagonal"),
                _decision(s.toward, 0, button, "skill_236_" + button),
                _decision(0, 0, "", "skill_release"),
            ))
            self.next_attack = s.frame + max(self.config.shot_interval, 5 * s.stride)
            return self.pending.popleft()
        return _decision(-s.toward, 0, "", "skill_reposition")

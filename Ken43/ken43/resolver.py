from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

from .data import FrameData, load_frame_data
from .frame_math import FrameCalculator
from .spacing import Posture, RecoveryMode, SpacingCalculator

Guard = Literal["standing", "crouching"]
StunKind = Literal["hit", "block"]


@dataclass
class FighterState:
    hp: int = 10000
    drive: float = 6.0
    burnout: bool = False
    burnout_pending_action: str | None = None
    posture: Posture = "standing"
    action_id: str | None = None
    action_frame: int = 0
    queued_action: str | None = None
    queued_source: str | None = None
    queued_recovery_mode: RecoveryMode = "hold2"
    action_source: str | None = None
    action_start_distance: float = 0.0
    action_recovery_mode: RecoveryMode = "hold2"
    stun_remaining: int = 0
    stun_kind: StunKind | None = None
    throw_invul_remaining: int = 0
    knockdown: bool = False
    airborne: bool = False
    blocking: bool = False
    action_connected: bool = False
    contact_action_frame: int | None = None
    contact_outcome: Literal["hit", "block", "throw"] | None = None
    throw_attempted: bool = False
    armor_hits_remaining: int = 0
    armor_absorbed_hit: bool = False
    pending_full_spacing: float | None = None
    combo_move_index: int = 0
    combo_penalty: float = 0.0
    special_stun: str | None = None
    special_stun_remaining: int = 0
    sacrifice_phase: str | None = None
    sacrifice_escape_until: int | None = None
    sacrifice_landing_remaining: int = 0


@dataclass
class CombatState:
    frame: int = 0
    distance: float = 0.0
    players: list[FighterState] = field(
        default_factory=lambda: [FighterState(), FighterState()]
    )
    terminal_reason: str | None = None

    def observation(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameInput:
    action: str | None = None
    guard: Guard | None = None
    cancel_to: str | None = None
    branch_to: str | None = None
    branch_window_override: bool = False
    reversal: str | None = None
    recovery_mode: RecoveryMode = "hold2"
    movement_delta: float = 0.0


@dataclass(frozen=True)
class FrameEvent:
    type: str
    actor: int | None = None
    target: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MovementProfile:
    """Known per-frame changes in player separation.

    Positive values increase distance and negative values reduce it. Empty by
    default because the current JSON mostly contains aggregate endpoints.
    """

    action_frame_deltas: Mapping[str, Mapping[int, float]] = field(default_factory=dict)

    def delta(self, action_id: str, action_frame: int) -> float:
        return float(self.action_frame_deltas.get(action_id, {}).get(action_frame, 0.0))


@dataclass(frozen=True)
class _ContactCandidate:
    actor: int
    target: int
    move_id: str
    action_frame: int
    active_index: int
    hit_level: str
    outcome: Literal["hit", "block", "throw"]


class FrameResolver:
    """Deterministic, symmetric, one-global-frame combat resolver."""

    def __init__(
        self,
        frame_data: FrameData | None = None,
        movement_profile: MovementProfile | None = None,
    ):
        self.data = frame_data or load_frame_data()
        self.calc = FrameCalculator(self.data)
        self.spacing = SpacingCalculator(self.data)
        self.movement_profile = movement_profile or MovementProfile()
        self._enabled_actions = set(self.calc.enabled_action_ids())
        self._direct_actions = set(self.calc.legal_action_ids())

    @property
    def direct_actions(self) -> tuple[str, ...]:
        return tuple(sorted(self._direct_actions))

    @property
    def enabled_actions(self) -> tuple[str, ...]:
        return tuple(sorted(self._enabled_actions))

    def combat_data_complete(self, move_id: str) -> bool:
        move = self.data.move(move_id)
        if move.get("hit_level") == "non_attack":
            return True
        effects = self.data.effects(move_id)
        drive = effects.get("drive")
        scaling = effects.get("scaling")
        if not isinstance(drive, list) or not isinstance(scaling, list):
            return False
        hit_stun_known = (
            move.get("stun", {}).get("hit") is not None
            or bool(move.get("knockdown"))
            or move.get("hit_level") == "throw"
            or move_id == "drive_impact"
        )
        if not hit_stun_known:
            return False
        if move.get("hit_level") == "throw":
            return drive[0] is not None and all(value is not None for value in scaling)
        return all(value is not None for value in drive) and all(
            value is not None for value in scaling
        )

    def direct_action_available(
        self,
        state: CombatState,
        actor: int,
        move_id: str,
        *,
        exact_combat: bool = False,
    ) -> bool:
        if move_id not in self._direct_actions or not self._is_actionable(state.players[actor]):
            return False
        move = self.data.move(move_id)
        if move_id == "drive_impact" and state.distance > float(
            move["supported_domain_max_distance"]
        ):
            return False
        effects = self.data.effects(move_id)
        drive = effects.get("drive", [None, None, None, move.get("drive_cost", 0)])
        cost = drive[3]
        if cost is None or state.players[actor].drive < float(cost):
            return False
        if exact_combat and not self.combat_data_complete(move_id):
            return False
        if (
            exact_combat
            and move.get("hit_level") == "throw"
            and self._counter_context(state.players[1 - actor]) == "punish_counter"
            and self.data.effects(move_id).get("drive", [None, None, None, None])[2]
            is None
        ):
            return False
        if exact_combat and move_id == "tatsu_L":
            threshold = float(self.data.combat_system["l_tatsu_full_hit_thresholds"]["raw"])
            if state.distance > threshold:
                return False
        return True

    def reversal_available(self, state: CombatState, actor: int) -> bool:
        fighter = state.players[actor]
        in_window = (
            fighter.stun_remaining > 0 and fighter.stun_kind == "block"
        ) or (not fighter.knockdown and fighter.throw_invul_remaining > 0)
        return in_window and fighter.drive >= 2.0

    def resolve_frame(
        self,
        state: CombatState,
        inputs: Sequence[FrameInput] | None = None,
    ) -> tuple[CombatState, list[FrameEvent]]:
        if inputs is None:
            inputs = (FrameInput(), FrameInput())
        if len(inputs) != 2:
            raise ValueError("resolve_frame requires exactly two player inputs")

        next_state = deepcopy(state)
        if len(next_state.players) != 2:
            raise ValueError("CombatState must contain exactly two players")
        next_state.frame += 1
        events: list[FrameEvent] = []

        self._advance_existing_actions(next_state)
        self._start_queued_actions(next_state, events)
        self._apply_postures(next_state, inputs)
        self._start_reversals(next_state, inputs, events)
        self._start_requested_actions(next_state, inputs, events)
        self._apply_branches(next_state, inputs, events)
        self._update_sacrifice_jumps(next_state, events)
        self._apply_frame_movement(next_state, inputs, events)

        candidates = self._collect_contacts(next_state, inputs, events)
        candidates = self._resolve_same_frame_conflicts(next_state, candidates, events)
        successful_contacts, interrupted = self._commit_contacts(
            next_state, inputs, candidates, events
        )
        self._interrupt_actions(next_state, interrupted, events)
        self._apply_contact_cancels(
            next_state, inputs, successful_contacts, events
        )
        self._finish_actions(next_state, events)
        self._tick_statuses(next_state, events)
        self._check_burnout(next_state, events)
        return next_state, events

    def _advance_existing_actions(self, state: CombatState) -> None:
        for fighter in state.players:
            if fighter.action_id is not None:
                fighter.action_frame += 1

    def _start_queued_actions(
        self, state: CombatState, events: list[FrameEvent]
    ) -> None:
        for actor, fighter in enumerate(state.players):
            if fighter.action_id is None and fighter.queued_action is not None:
                action_id = fighter.queued_action
                fighter.queued_action = None
                source = fighter.queued_source
                recovery_mode = fighter.queued_recovery_mode
                fighter.queued_source = None
                fighter.queued_recovery_mode = "hold2"
                self._start_action(
                    state,
                    actor,
                    action_id,
                    1,
                    "ActionStarted",
                    events,
                    source=source,
                    recovery_mode=recovery_mode,
                )

    def _apply_postures(
        self, state: CombatState, inputs: Sequence[FrameInput]
    ) -> None:
        for fighter, frame_input in zip(state.players, inputs):
            if frame_input.guard is not None:
                fighter.posture = frame_input.guard

    def _start_requested_actions(
        self,
        state: CombatState,
        inputs: Sequence[FrameInput],
        events: list[FrameEvent],
    ) -> None:
        for actor, frame_input in enumerate(inputs):
            if frame_input.action is None:
                continue
            fighter = state.players[actor]
            if frame_input.action not in self._direct_actions:
                events.append(
                    FrameEvent(
                        "InputRejected",
                        actor,
                        data={
                            "action": frame_input.action,
                            "reason": "not_directly_selectable",
                        },
                    )
                )
                continue
            if not self._is_actionable(fighter):
                events.append(
                    FrameEvent(
                        "InputRejected",
                        actor,
                        data={"action": frame_input.action, "reason": "not_actionable"},
                    )
                )
                continue
            self._start_action(
                state,
                actor,
                frame_input.action,
                1,
                "ActionStarted",
                events,
                recovery_mode=frame_input.recovery_mode,
            )

    def _start_reversals(
        self,
        state: CombatState,
        inputs: Sequence[FrameInput],
        events: list[FrameEvent],
    ) -> None:
        for actor, frame_input in enumerate(inputs):
            if frame_input.reversal != "drive_reversal":
                continue
            fighter = state.players[actor]
            if fighter.stun_remaining > 0 and fighter.stun_kind == "block":
                action_id = "drive_reversal_block"
            elif not fighter.knockdown and fighter.throw_invul_remaining > 0:
                action_id = "drive_reversal_wakeup"
            else:
                events.append(
                    FrameEvent(
                        "InputRejected",
                        actor,
                        data={"reversal": "drive_reversal", "reason": "outside_reversal_window"},
                    )
                )
                continue
            cost = float(self.data.effects(action_id).get("drive", [0, 0, 0, 2.0])[3])
            if fighter.drive < cost:
                events.append(
                    FrameEvent(
                        "InputRejected",
                        actor,
                        data={"reversal": "drive_reversal", "reason": "insufficient_drive"},
                    )
                )
                continue
            self._clear_action(fighter)
            fighter.stun_remaining = 0
            fighter.stun_kind = None
            fighter.blocking = False
            self._start_action(
                state,
                actor,
                action_id,
                1,
                "ReversalStarted",
                events,
                source="blockstun" if action_id.endswith("block") else "wakeup",
            )

    def _start_action(
        self,
        state: CombatState,
        actor: int,
        action_id: str,
        action_frame: int,
        event_type: str,
        events: list[FrameEvent],
        *,
        source: str | None = None,
        recovery_mode: RecoveryMode = "hold2",
    ) -> bool:
        if action_id not in self._enabled_actions:
            events.append(
                FrameEvent(
                    "InputRejected",
                    actor,
                    data={"action": action_id, "reason": "masked_or_unknown"},
                )
            )
            return False
        move = self.data.move(action_id)
        fighter = state.players[actor]
        effects = self.data.effects(action_id)
        drive_values = effects.get("drive", [None, None, None, move.get("drive_cost", 0)])
        drive_cost = float(drive_values[3] if drive_values[3] is not None else 0.0)
        drive_enabled = bool(
            self.data.system_rules.get("drive_resource_enabled", False)
        )
        if drive_enabled and fighter.drive < drive_cost:
            events.append(
                FrameEvent(
                    "InputRejected",
                    actor,
                    data={"action": action_id, "reason": "insufficient_drive"},
                )
            )
            return False

        if drive_enabled:
            fighter.drive = max(0.0, fighter.drive - drive_cost)
            if fighter.drive == 0.0 and action_id in {
                "drive_reversal_block",
                "drive_reversal_wakeup",
            }:
                fighter.burnout_pending_action = action_id
        fighter.action_id = action_id
        fighter.action_frame = action_frame
        fighter.action_source = source
        fighter.action_start_distance = state.distance
        fighter.action_recovery_mode = recovery_mode
        fighter.action_connected = False
        fighter.contact_action_frame = None
        fighter.contact_outcome = None
        fighter.throw_attempted = False
        fighter.pending_full_spacing = None
        fighter.airborne = move.get("kind") == "air_normal"
        fighter.armor_hits_remaining = int(move.get("armor", {}).get("hits", 0))
        fighter.armor_absorbed_hit = False
        fighter.sacrifice_phase = "startup" if action_id == "sacrifice_jump" else None
        events.append(
            FrameEvent(
                event_type,
                actor,
                data={
                    "action": action_id,
                    "action_frame": action_frame,
                    "drive_cost": drive_cost if drive_enabled else 0.0,
                    "recovery_mode": recovery_mode,
                },
            )
        )
        return True

    def _apply_branches(
        self,
        state: CombatState,
        inputs: Sequence[FrameInput],
        events: list[FrameEvent],
    ) -> None:
        for actor, frame_input in enumerate(inputs):
            if frame_input.branch_to is None:
                continue
            fighter = state.players[actor]
            if fighter.action_id is None:
                events.append(
                    FrameEvent(
                        "InputRejected",
                        actor,
                        data={"branch": frame_input.branch_to, "reason": "no_parent_action"},
                    )
                )
                continue
            child = self._legal_branch_child(
                fighter.action_id, fighter.action_frame, frame_input.branch_to
            )
            if child is None and frame_input.branch_window_override:
                for branch_name, branch in self.data.move(fighter.action_id).get("branches", {}).items():
                    candidate = branch.get("child", branch.get("result_action"))
                    if frame_input.branch_to in (branch_name, candidate):
                        child = candidate
                        break
            if child is None:
                events.append(
                    FrameEvent(
                        "InputRejected",
                        actor,
                        data={"branch": frame_input.branch_to, "reason": "outside_branch_window"},
                    )
                )
                continue
            if child not in self._enabled_actions:
                events.append(
                    FrameEvent(
                        "InputRejected",
                        actor,
                        data={
                            "branch": frame_input.branch_to,
                            "reason": "masked_or_unknown_child",
                        },
                    )
                )
                continue

            parent_id = fighter.action_id
            parent_frame = fighter.action_frame
            initial_frame = self._branch_initial_frame(
                parent_id, parent_frame, child
            )
            self._clear_action(fighter)
            if initial_frame <= 0:
                fighter.queued_action = child
                fighter.queued_source = parent_id
                events.append(
                    FrameEvent(
                        "BranchQueued",
                        actor,
                        data={"parent": parent_id, "child": child},
                    )
                )
            else:
                self._start_action(
                    state,
                    actor,
                    child,
                    initial_frame,
                    "BranchStarted",
                    events,
                    source=parent_id,
                )

    def _legal_branch_child(
        self, parent_id: str, parent_frame: int, requested: str
    ) -> str | None:
        parent = self.data.move(parent_id)
        for branch_name, branch in parent.get("branches", {}).items():
            child = branch.get("child", branch.get("result_action"))
            if requested not in (branch_name, child):
                continue
            if "input_window" in branch:
                start, end = map(int, branch["input_window"])
                return child if start <= parent_frame <= end else None
            if "input_frame" in branch:
                return child if parent_frame == int(branch["input_frame"]) else None
        return None

    def _branch_initial_frame(
        self, parent_id: str, parent_frame: int, child_id: str
    ) -> int:
        child = self.data.move(child_id)
        startup = int(child.get("startup", child.get("startup_from_branch", 1)))
        desired_first_active: int | None = None

        if "first_active_from_quick_dash_start" in child:
            desired_first_active = int(child["first_active_from_quick_dash_start"])
        parent = self.data.move(parent_id)
        empirical = parent.get("empirical_branch_timing", {})
        if child_id == "kazekama" and "fastest_LK_followup_first_active_root_frame" in empirical:
            desired_first_active = int(
                empirical["fastest_LK_followup_first_active_root_frame"]
            )

        if desired_first_active is None:
            return 1
        return startup - (desired_first_active - parent_frame)

    def _update_sacrifice_jumps(
        self, state: CombatState, events: list[FrameEvent]
    ) -> None:
        for actor, fighter in enumerate(state.players):
            if fighter.action_id != "sacrifice_jump":
                continue
            if fighter.sacrifice_phase == "startup" and fighter.action_frame >= 5:
                opponent = state.players[1 - actor]
                remaining = 0
                if opponent.action_id is not None:
                    active = self.calc.active_frames(opponent.action_id)
                    if active and max(active) >= opponent.action_frame:
                        remaining = max(active) - opponent.action_frame
                fighter.sacrifice_phase = "airborne_escape"
                fighter.airborne = True
                fighter.sacrifice_escape_until = state.frame + remaining
                events.append(
                    FrameEvent(
                        "SacrificeJumpTakeoff",
                        actor,
                        data={"escape_until_frame": fighter.sacrifice_escape_until},
                    )
                )
            if (
                fighter.sacrifice_phase == "airborne_escape"
                and fighter.sacrifice_escape_until is not None
                and state.frame > fighter.sacrifice_escape_until
            ):
                fighter.sacrifice_phase = "landing"
                fighter.airborne = False
                # The transition frame precedes the three complete recovery frames.
                fighter.sacrifice_landing_remaining = 4
                events.append(FrameEvent("SacrificeJumpLanding", actor))

    def _apply_frame_movement(
        self,
        state: CombatState,
        inputs: Sequence[FrameInput],
        events: list[FrameEvent],
    ) -> None:
        total_delta = 0.0
        components: list[dict[str, Any]] = []
        for actor, (fighter, frame_input) in enumerate(zip(state.players, inputs)):
            if (
                frame_input.movement_delta
                and fighter.action_id is None
                and fighter.stun_remaining == 0
                and not fighter.knockdown
            ):
                delta = float(frame_input.movement_delta)
                total_delta += delta
                components.append({"actor": actor, "source": "input", "delta": delta})
            if fighter.action_id is not None:
                delta = self.movement_profile.delta(
                    fighter.action_id, fighter.action_frame
                )
                if delta:
                    total_delta += delta
                    components.append(
                        {
                            "actor": actor,
                            "source": "profile",
                            "action": fighter.action_id,
                            "action_frame": fighter.action_frame,
                            "delta": delta,
                        }
                    )
        if total_delta:
            before = state.distance
            state.distance = max(0.0, state.distance + total_delta)
            events.append(
                FrameEvent(
                    "Movement",
                    data={
                        "mode": "per_frame",
                        "before": before,
                        "after": state.distance,
                        "components": components,
                    },
                )
            )

    def _collect_contacts(
        self,
        state: CombatState,
        inputs: Sequence[FrameInput],
        events: list[FrameEvent],
    ) -> list[_ContactCandidate]:
        candidates: list[_ContactCandidate] = []
        for actor, fighter in enumerate(state.players):
            if fighter.action_id is None or fighter.action_connected:
                continue
            move_id = fighter.action_id
            move = self.data.move(move_id)
            active_index = self._active_index(move_id, fighter.action_frame)
            if active_index is None:
                continue
            target = 1 - actor
            defender = state.players[target]
            hit_level = str(move.get("hit_level", "non_attack"))
            events.append(
                FrameEvent(
                    "ActiveFrame",
                    actor,
                    target,
                    {
                        "action": move_id,
                        "action_frame": fighter.action_frame,
                        "active_index": active_index,
                    },
                )
            )
            if hit_level == "non_attack" or defender.knockdown:
                continue

            if not self._in_range(move_id, state.distance, defender.posture):
                events.append(
                    FrameEvent(
                        "RangeMiss",
                        actor,
                        target,
                        {
                            "action": move_id,
                            "action_frame": fighter.action_frame,
                            "distance": state.distance,
                        },
                    )
                )
                if hit_level == "throw" and not fighter.throw_attempted:
                    fighter.throw_attempted = True
                    fighter.action_connected = True
                    events.append(
                        FrameEvent("ThrowWhiff", actor, target, {"reason": "out_of_range"})
                    )
                continue

            if hit_level == "throw":
                if fighter.throw_attempted:
                    continue
                fighter.throw_attempted = True
                if not self._throwable(defender):
                    fighter.action_connected = True
                    events.append(
                        FrameEvent(
                            "ThrowWhiff",
                            actor,
                            target,
                            {"reason": "target_unthrowable"},
                        )
                    )
                    continue
                outcome: Literal["hit", "block", "throw"] = "throw"
            else:
                if self._strike_invulnerable(defender, fighter):
                    events.append(
                        FrameEvent(
                            "Invulnerable",
                            target,
                            actor,
                            {"against": move_id},
                        )
                    )
                    continue
                guard = self._available_guard(defender, inputs[target].guard)
                outcome = "block" if self._blocks(hit_level, guard) else "hit"

            candidates.append(
                _ContactCandidate(
                    actor,
                    target,
                    move_id,
                    fighter.action_frame,
                    active_index,
                    hit_level,
                    outcome,
                )
            )
        return candidates

    def _resolve_same_frame_conflicts(
        self,
        state: CombatState,
        candidates: list[_ContactCandidate],
        events: list[FrameEvent],
    ) -> list[_ContactCandidate]:
        if len(candidates) != 2:
            return candidates
        first, second = candidates
        if first.target != second.actor or second.target != first.actor:
            return candidates
        first_throw = first.hit_level == "throw"
        second_throw = second.hit_level == "throw"
        if first_throw and second_throw:
            state.players[first.actor].action_connected = True
            state.players[second.actor].action_connected = True
            gain = float(self.data.combat_system.get("throw_tech_drive_gain_each", 0.0))
            if self.data.system_rules.get("drive_resource_enabled", False):
                maximum = float(self.data.combat_system.get("drive_max", 6.0))
                for fighter in state.players:
                    fighter.drive = min(maximum, fighter.drive + gain)
            events.append(
                FrameEvent(
                    "ThrowClash",
                    data={"actors": [0, 1], "drive_gain_each": gain},
                )
            )
            return []
        if first_throw != second_throw:
            throw_candidate = first if first_throw else second
            strike_candidate = second if first_throw else first
            state.players[throw_candidate.actor].action_connected = True
            events.append(
                FrameEvent(
                    "ThrowLostToStrike",
                    throw_candidate.actor,
                    strike_candidate.actor,
                )
            )
            return [strike_candidate]
        return candidates

    def _commit_contacts(
        self,
        state: CombatState,
        inputs: Sequence[FrameInput],
        candidates: list[_ContactCandidate],
        events: list[FrameEvent],
    ) -> tuple[set[int], set[int]]:
        successful_actors: set[int] = set()
        interrupted: set[int] = set()
        damage = [0, 0]
        drive_delta = [0.0, 0.0]
        stuns: list[tuple[int, StunKind, int, Guard | None]] = []
        knockdowns: set[int] = set()
        special_stuns: list[tuple[int, str, int]] = []
        immediate_spacing_targets: list[float] = []

        for candidate in candidates:
            attacker = state.players[candidate.actor]
            defender = state.players[candidate.target]
            move = self.data.move(candidate.move_id)

            hit_count, known_damage = self._physical_hit_count(attacker, candidate.move_id)
            if candidate.hit_level != "throw" and self._armor_active(defender):
                counter_di = (
                    candidate.move_id == "drive_impact"
                    and defender.action_id == "drive_impact"
                )
                if counter_di:
                    defender.action_connected = True
                    defender.contact_action_frame = defender.action_frame
                    defender.contact_outcome = "counter_di"
                    successful_actors.add(candidate.target)
                    interrupted.add(candidate.actor)
                    events.append(
                        FrameEvent(
                            "CounterDI",
                            candidate.target,
                            candidate.actor,
                            {"action": "drive_impact"},
                        )
                    )
                    continue
                elif move.get("armor_breaker"):
                    absorbed_hits = defender.armor_hits_remaining
                    defender.armor_hits_remaining = 0
                    events.append(
                        FrameEvent(
                            "ArmorBreak",
                            candidate.actor,
                            candidate.target,
                            {
                                "action": candidate.move_id,
                                "hit_count": hit_count,
                                "absorbed_hits": absorbed_hits,
                                "armor_breaker": True,
                            },
                        )
                    )
                elif hit_count <= defender.armor_hits_remaining:
                    defender.armor_hits_remaining -= hit_count
                    defender.armor_absorbed_hit = True
                    attacker.action_connected = True
                    events.append(
                        FrameEvent(
                            "ArmorAbsorb",
                            candidate.target,
                            candidate.actor,
                            {
                                "action": candidate.move_id,
                                "hit_count": hit_count,
                                "absorbed_hits": hit_count,
                                "armor_hits_remaining": defender.armor_hits_remaining,
                            },
                        )
                    )
                    continue
                else:
                    absorbed_hits = defender.armor_hits_remaining
                    defender.armor_hits_remaining = 0
                    events.append(
                        FrameEvent(
                            "ArmorBreak",
                            candidate.actor,
                            candidate.target,
                            {
                                "action": candidate.move_id,
                                "hit_count": hit_count,
                                "absorbed_hits": absorbed_hits,
                                "penetrating_hits": hit_count - absorbed_hits,
                            },
                        )
                    )

            attacker.action_connected = True
            attacker.contact_action_frame = candidate.action_frame
            attacker.contact_outcome = candidate.outcome
            successful_actors.add(candidate.actor)
            interrupted.add(candidate.target)

            if candidate.outcome in ("hit", "throw"):
                counter_context = self._counter_context(defender)
                armor_conversion = (
                    candidate.move_id == "drive_impact"
                    and attacker.armor_absorbed_hit
                    and candidate.outcome == "hit"
                )
                amount, scale = self._scaled_damage(
                    attacker,
                    defender,
                    candidate.move_id,
                    counter_context,
                    armor_conversion,
                    known_damage,
                )
                if not known_damage:
                    events.append(
                        FrameEvent(
                            "DataIncomplete",
                            candidate.actor,
                            candidate.target,
                            {"action": candidate.move_id, "field": "single_hit_damage"},
                        )
                    )
                damage[candidate.target] += amount
                effects = self.data.effects(candidate.move_id)
                drive_values = effects.get("drive", [None, None, None, 0.0])
                gain = drive_values[0]
                drive_damage = drive_values[2] if counter_context == "punish_counter" else 0.0
                if armor_conversion:
                    drive_damage = self.data.combat_system["di"]["armor_conversion_drive_damage"]
                if gain is None or (counter_context == "punish_counter" and drive_damage is None):
                    events.append(
                        FrameEvent(
                            "DataIncomplete",
                            candidate.actor,
                            candidate.target,
                            {"action": candidate.move_id, "field": "drive_on_hit"},
                        )
                    )
                else:
                    drive_delta[candidate.actor] += float(gain)
                    drive_delta[candidate.target] -= float(drive_damage or 0.0)

                is_di = candidate.move_id == "drive_impact"
                if is_di and (armor_conversion or counter_context == "punish_counter"):
                    remaining = self._action_total(candidate.move_id) - candidate.action_frame
                    special_stuns.append((candidate.target, "crumple", remaining + 66))
                elif is_di:
                    remaining = self._action_total(candidate.move_id) - candidate.action_frame
                    advantage = int(self.data.combat_system["di"]["wall_splat_hit_advantage"])
                    special_stuns.append((candidate.target, "wall_splat", remaining + advantage + 1))
                elif move.get("knockdown") or candidate.outcome == "throw":
                    knockdowns.add(candidate.target)
                stun = self._contact_stun(candidate, "hit")
                if counter_context == "counter_hit":
                    stun += int(self.data.combat_system["counter_hit_stun_bonus"])
                elif counter_context == "punish_counter":
                    stun += int(self.data.combat_system["punish_counter_stun_bonus"])
                if (
                    stun > 0
                    and candidate.target not in knockdowns
                    and not is_di
                ):
                    stuns.append((candidate.target, "hit", stun, None))
                events.append(
                    FrameEvent(
                        "Throw" if candidate.outcome == "throw" else "Hit",
                        candidate.actor,
                        candidate.target,
                        {
                            "action": candidate.move_id,
                            "active_index": candidate.active_index,
                            "hit_count": hit_count,
                            "damage": amount,
                            "damage_scale": scale,
                            "stun": stun,
                            "counter_context": counter_context,
                            "armor_conversion": armor_conversion,
                            "drive_gain_attacker": float(gain or 0.0),
                            "drive_damage_defender": float(drive_damage or 0.0),
                        },
                    )
                )
                if defender.special_stun is not None:
                    defender.special_stun = None
                    defender.special_stun_remaining = 0
            else:
                guard = inputs[candidate.target].guard or defender.posture
                stun = self._contact_stun(candidate, "block")
                effects = self.data.effects(candidate.move_id)
                drive_values = effects.get("drive", [None, None, None, 0.0])
                drive_damage = drive_values[1]
                if drive_damage is None:
                    events.append(
                        FrameEvent(
                            "DataIncomplete",
                            candidate.actor,
                            candidate.target,
                            {"action": candidate.move_id, "field": "drive_on_block"},
                        )
                    )
                else:
                    drive_delta[candidate.target] -= float(drive_damage)
                if candidate.move_id == "drive_impact":
                    remaining = self._action_total(candidate.move_id) - candidate.action_frame
                    advantage = int(self.data.combat_system["di"]["wall_splat_block_advantage"])
                    special_stuns.append((candidate.target, "wall_splat", remaining + advantage + 1))
                else:
                    stuns.append((candidate.target, "block", stun, guard))
                immediate = self._prepare_block_spacing(
                    state, candidate.actor, candidate.move_id
                )
                if immediate is not None:
                    immediate_spacing_targets.append(immediate)
                events.append(
                    FrameEvent(
                        "Block",
                        candidate.actor,
                        candidate.target,
                        {
                            "action": candidate.move_id,
                            "active_index": candidate.active_index,
                            "hit_count": hit_count,
                            "stun": stun,
                            "guard": guard,
                            "drive_damage_defender": float(drive_damage or 0.0),
                        },
                    )
                )

        for target, amount in enumerate(damage):
            if amount:
                state.players[target].hp = max(0, state.players[target].hp - amount)
        maximum_drive = float(self.data.combat_system.get("drive_max", 6.0))
        if self.data.system_rules.get("drive_resource_enabled", False):
            for target, delta in enumerate(drive_delta):
                if delta:
                    fighter = state.players[target]
                    before = fighter.drive
                    fighter.drive = min(maximum_drive, max(0.0, fighter.drive + delta))
                    events.append(
                        FrameEvent(
                            "DriveChanged",
                            target,
                            data={"before": before, "after": fighter.drive, "delta": delta},
                        )
                    )
        for target, kind, amount, guard in stuns:
            fighter = state.players[target]
            fighter.stun_remaining = max(fighter.stun_remaining, amount)
            fighter.stun_kind = kind
            fighter.blocking = kind == "block"
            if guard is not None:
                fighter.posture = guard
        for target in knockdowns:
            fighter = state.players[target]
            fighter.knockdown = True
            fighter.stun_remaining = 0
            fighter.stun_kind = None
            fighter.blocking = False
        for target, kind, amount in special_stuns:
            fighter = state.players[target]
            fighter.special_stun = kind
            fighter.special_stun_remaining = max(fighter.special_stun_remaining, amount)
            fighter.stun_remaining = 0
            fighter.stun_kind = None
            fighter.blocking = False
            events.append(
                FrameEvent(
                    "SpecialStun",
                    target,
                    data={"kind": kind, "remaining": fighter.special_stun_remaining},
                )
            )

        if immediate_spacing_targets:
            before = state.distance
            state.distance = max([state.distance, *immediate_spacing_targets])
            if state.distance != before:
                events.append(
                    FrameEvent(
                        "Movement",
                        data={
                            "mode": "aggregate_contact_fallback",
                            "before": before,
                            "after": state.distance,
                        },
                    )
                )
        return successful_actors, interrupted

    def _physical_hit_count(
        self, attacker: FighterState, move_id: str
    ) -> tuple[int, bool]:
        effects = self.data.effects(move_id)
        if move_id != "tatsu_L":
            return int(effects.get("physical_hits", self.calc.simplified_hit_count(move_id))), True
        route = attacker.action_source or "raw"
        threshold = self.data.combat_system["l_tatsu_full_hit_thresholds"].get(route)
        if threshold is not None and attacker.action_start_distance <= float(threshold):
            return 2, True
        return 1, False

    def _counter_context(self, defender: FighterState) -> str:
        if defender.action_id is None:
            return "normal"
        active = self.calc.active_frames(defender.action_id)
        if not active:
            return "normal"
        if defender.action_frame < min(active):
            return "counter_hit"
        if defender.action_frame <= max(active):
            return "counter_hit"
        return "punish_counter"

    def _scaled_damage(
        self,
        attacker: FighterState,
        defender: FighterState,
        move_id: str,
        counter_context: str,
        armor_conversion: bool,
        known_damage: bool,
    ) -> tuple[int, float]:
        if not known_damage:
            return 0, 0.0
        move = self.data.move(move_id)
        effects = self.data.effects(move_id)
        base = int(move.get("damage", 0))
        is_throw = move.get("hit_level") == "throw"
        continuing = (
            defender.stun_remaining > 0 and defender.stun_kind == "hit"
        ) or defender.special_stun in {"wall_splat", "crumple"}
        if not continuing or is_throw:
            attacker.combo_move_index = 0
            attacker.combo_penalty = 0.0
        attacker.combo_move_index += 1
        index = attacker.combo_move_index
        base_scale = 1.0 if index <= 2 else max(0.10, 1.10 - 0.10 * index)
        scaling = effects.get("scaling", [0.0, 0.0, 0.0])
        immediate = float(scaling[2] or 0.0)
        scale = max(
            float(self.data.combat_system.get("combo_floor", 0.10)),
            base_scale - attacker.combo_penalty - immediate,
        )
        multiplier = 1.0
        if (
            not is_throw
            and (counter_context in {"counter_hit", "punish_counter"} or armor_conversion)
        ):
            multiplier = float(self.data.combat_system["punish_counter_multiplier"])
        override = effects.get("punish_counter_damage_override")
        if is_throw and counter_context == "punish_counter" and override is not None:
            amount = int(override)
        else:
            amount = int(base * scale * multiplier)
        if index == 1:
            attacker.combo_penalty += float(scaling[0] or 0.0)
        attacker.combo_penalty += float(scaling[1] or 0.0)
        return amount, scale

    def _interrupt_actions(
        self,
        state: CombatState,
        interrupted: set[int],
        events: list[FrameEvent],
    ) -> None:
        for target in interrupted:
            fighter = state.players[target]
            if fighter.action_id is not None:
                action = fighter.action_id
                self._clear_action(fighter)
                events.append(
                    FrameEvent("ActionInterrupted", target, data={"action": action})
                )

    def _apply_contact_cancels(
        self,
        state: CombatState,
        inputs: Sequence[FrameInput],
        successful_actors: set[int],
        events: list[FrameEvent],
    ) -> None:
        for actor, frame_input in enumerate(inputs):
            if frame_input.cancel_to is None:
                continue
            fighter = state.players[actor]
            parent = fighter.action_id
            has_prior_contact = fighter.contact_action_frame is not None
            if (actor not in successful_actors and not has_prior_contact) or parent is None:
                events.append(
                    FrameEvent(
                        "InputRejected",
                        actor,
                        data={"cancel": frame_input.cancel_to, "reason": "no_contact"},
                    )
                )
                continue
            delay = fighter.action_frame - int(fighter.contact_action_frame)
            if not self._cancel_allowed(
                parent, frame_input.cancel_to, delay, fighter.action_frame
            ):
                events.append(
                    FrameEvent(
                        "InputRejected",
                        actor,
                        data={"cancel": frame_input.cancel_to, "reason": "illegal_cancel"},
                    )
                )
                continue
            child = frame_input.cancel_to
            recovery_mode = frame_input.recovery_mode
            fighter.pending_full_spacing = None
            self._clear_action(fighter)
            fighter.queued_action = child
            fighter.queued_source = parent
            fighter.queued_recovery_mode = recovery_mode
            events.append(
                FrameEvent(
                    "CancelQueued",
                    actor,
                    data={"parent": parent, "child": child},
                )
            )

    def _cancel_allowed(
        self,
        parent_id: str,
        child_id: str,
        delay: int,
        current_action_frame: int,
    ) -> bool:
        if delay < 0 or child_id not in self._enabled_actions:
            return False
        cancel = self.data.move(parent_id).get("cancel", {})
        chain = cancel.get("chain", {})
        if child_id in chain.get("targets", []):
            if delay == 0:
                return True
            if not chain.get("delayable"):
                return False
            if chain.get("delay_policy") == "within_remaining_active_frames":
                active = self.calc.active_frames(parent_id)
                return bool(active) and current_action_frame <= max(active)
            return True
        target_combo = cancel.get("target_combo", {})
        if child_id in target_combo.get("targets", []):
            return delay == 0 or bool(target_combo.get("delayable"))
        special = cancel.get("special", {})
        if not special.get("enabled"):
            return False
        if delay not in [int(value) for value in special.get("default_delay_frames", [0])]:
            return False
        if child_id in special.get("also_allows", []):
            return True
        targets = special.get("targets")
        if isinstance(targets, list):
            return child_id in targets
        if targets == "all_enabled_specials":
            kind = self.data.move(child_id).get("kind")
            return kind in {
                "special",
                "projectile_special",
                "jinrai_root",
                "system_transition",
                "system",
            }
        return False

    def _finish_actions(
        self, state: CombatState, events: list[FrameEvent]
    ) -> None:
        for actor, fighter in enumerate(state.players):
            if fighter.action_id is None:
                continue
            if fighter.action_id == "sacrifice_jump":
                continue
            total = self._action_total(fighter.action_id)
            move = self.data.move(fighter.action_id)
            if not fighter.action_connected:
                total += int(move.get("extra_recovery_on_whiff_or_armor", 0))
            elif fighter.contact_outcome == "block":
                total += int(move.get("extra_recovery_on_block", 0))
            if fighter.action_frame < total:
                continue
            action = fighter.action_id
            if (
                not fighter.action_connected
                and move.get("hit_level") not in (None, "non_attack")
            ):
                events.append(FrameEvent("Whiff", actor, data={"action": action}))
                before = state.distance
                after, displacement = self.spacing.spacing_after_whiff(
                    action, before, fighter.action_recovery_mode
                )
                state.distance = after
                if displacement:
                    events.append(
                        FrameEvent(
                            "Movement",
                            actor,
                            data={
                                "mode": "aggregate_whiff",
                                "action": action,
                                "before": before,
                                "after": after,
                                "forward_displacement": displacement,
                            },
                        )
                    )
            if fighter.pending_full_spacing is not None:
                before = state.distance
                state.distance = max(state.distance, fighter.pending_full_spacing)
                if state.distance != before:
                    events.append(
                        FrameEvent(
                            "Movement",
                            actor,
                            data={
                                "mode": "aggregate_recovery_fallback",
                                "action": action,
                                "before": before,
                                "after": state.distance,
                            },
                        )
                    )
            self._clear_action(fighter)
            events.append(FrameEvent("ActionEnded", actor, data={"action": action}))

    def _tick_statuses(
        self, state: CombatState, events: list[FrameEvent]
    ) -> None:
        for actor, fighter in enumerate(state.players):
            if fighter.throw_invul_remaining > 0:
                fighter.throw_invul_remaining -= 1
            if fighter.special_stun_remaining > 0:
                fighter.special_stun_remaining -= 1
                if fighter.special_stun_remaining == 0:
                    kind = fighter.special_stun
                    fighter.special_stun = None
                    events.append(FrameEvent("SpecialStunEnded", actor, data={"kind": kind}))
            if fighter.sacrifice_phase == "landing":
                fighter.sacrifice_landing_remaining -= 1
                if fighter.sacrifice_landing_remaining <= 0:
                    self._clear_action(fighter)
                    events.append(FrameEvent("ActionEnded", actor, data={"action": "sacrifice_jump"}))
            if fighter.stun_remaining > 0:
                kind = fighter.stun_kind
                fighter.stun_remaining -= 1
            else:
                kind = None
            if fighter.stun_remaining == 0 and kind is not None:
                key = (
                    "post_hitstun_throw_invulnerability"
                    if kind == "hit"
                    else "post_blockstun_throw_invulnerability"
                )
                fighter.throw_invul_remaining = max(
                    fighter.throw_invul_remaining,
                    int(self.data.system_rules[key]),
                )
                fighter.stun_kind = None
                fighter.blocking = False
                opponent = state.players[1 - actor]
                opponent.combo_move_index = 0
                opponent.combo_penalty = 0.0
                events.append(FrameEvent("StunEnded", actor, data={"kind": kind}))

    def _check_burnout(
        self, state: CombatState, events: list[FrameEvent]
    ) -> None:
        newly_burned: list[int] = []
        for actor, fighter in enumerate(state.players):
            if fighter.burnout_pending_action is not None:
                if fighter.action_id == fighter.burnout_pending_action and not fighter.action_connected:
                    continue
                fighter.drive = 0.0
                fighter.burnout = True
                newly_burned.append(actor)
                outcome = fighter.contact_outcome
                state.terminal_reason = (
                    "burnout_neutral_reset"
                    if outcome == "hit"
                    else "burnout_failed_escape"
                )
                fighter.burnout_pending_action = None
                events.append(FrameEvent("Burnout", actor))
                continue
            if fighter.drive <= 0.0 and not fighter.burnout:
                fighter.drive = 0.0
                fighter.burnout = True
                newly_burned.append(actor)
                events.append(FrameEvent("Burnout", actor))
        if not newly_burned:
            return
        if len(newly_burned) == 2:
            state.terminal_reason = "double_burnout"
            return
        if state.terminal_reason is not None:
            return
        actor = newly_burned[0]
        fighter = state.players[actor]
        if fighter.action_id in {"drive_reversal_block", "drive_reversal_wakeup"}:
            state.terminal_reason = (
                "burnout_neutral_reset"
                if fighter.action_connected
                else "burnout_failed_escape"
            )
        else:
            state.terminal_reason = "burnout_close_pressure"

    def _active_index(self, move_id: str, action_frame: int) -> int | None:
        move = self.data.move(move_id)
        frames = self.calc.active_frames(move_id)
        if not frames and move.get("kind") == "projectile_special":
            frames = [int(move["projectile_spawn_frame"])]
        for index, frame in enumerate(frames, start=1):
            if frame == action_frame:
                return index
        return None

    def _in_range(self, move_id: str, distance: float, posture: Posture) -> bool:
        move = self.data.move(move_id)
        if move_id == "drive_impact":
            return float(distance) <= float(move["supported_domain_max_distance"])
        measurement = self.data.system_rules.get("range_measurements", {}).get(move_id, {})
        if (
            isinstance(move.get("attack_reach"), (int, float))
            or isinstance(move.get("effective_range"), dict)
            or isinstance(measurement.get("standing_first_whiff"), (int, float))
        ):
            return self.spacing.in_range(move_id, distance, posture)
        return True

    def _blocks(self, hit_level: str, guard: Guard | None) -> bool:
        if guard is None:
            return False
        if hit_level == "low":
            return guard == "crouching"
        if hit_level == "overhead":
            return guard == "standing"
        return hit_level == "mid"

    def _available_guard(
        self, fighter: FighterState, requested_guard: Guard | None
    ) -> Guard | None:
        if fighter.action_id is not None:
            return None
        if fighter.special_stun_remaining > 0:
            return None
        if fighter.stun_remaining > 0 and fighter.stun_kind == "hit":
            return None
        if requested_guard is not None:
            return requested_guard
        if fighter.blocking and fighter.stun_kind == "block":
            return fighter.posture
        return None

    def _throwable(self, fighter: FighterState) -> bool:
        return not (
            fighter.throw_invul_remaining > 0
            or fighter.stun_remaining > 0
            or fighter.special_stun_remaining > 0
            or fighter.knockdown
            or fighter.airborne
            or fighter.action_id == "sacrifice_jump"
        )

    def _strike_invulnerable(
        self, defender: FighterState, attacker: FighterState
    ) -> bool:
        if (
            defender.action_id == "sacrifice_jump"
            and defender.sacrifice_phase == "airborne_escape"
        ):
            return True
        if defender.action_id is None:
            return False
        move = self.data.move(defender.action_id)
        for window in move.get("invulnerability", []):
            start, end = map(int, window["frames"])
            if not start <= defender.action_frame <= end:
                continue
            kind = window["type"]
            if kind == "full" or kind == "strike":
                return True
            if kind == "air_strike" and attacker.airborne:
                return True
        return False

    def active_invulnerability_types(self, fighter: FighterState) -> tuple[str, ...]:
        if fighter.action_id is None:
            return ()
        active: list[str] = []
        if fighter.action_id == "sacrifice_jump":
            if fighter.sacrifice_phase == "airborne_escape":
                active.append("strike")
            if fighter.sacrifice_phase in {"startup", "airborne_escape"}:
                active.append("throw")
        for window in self.data.move(fighter.action_id).get("invulnerability", []):
            start, end = map(int, window["frames"])
            if start <= fighter.action_frame <= end:
                active.append(str(window["type"]))
        return tuple(active)

    def _armor_active(self, fighter: FighterState) -> bool:
        if fighter.action_id is None or fighter.armor_hits_remaining <= 0:
            return False
        armor = self.data.move(fighter.action_id).get("armor")
        if not armor:
            return False
        start, end = map(int, armor["frames"])
        return start <= fighter.action_frame <= end

    def _contact_stun(
        self, candidate: _ContactCandidate, outcome: StunKind
    ) -> int:
        move = self.data.move(candidate.move_id)
        value = move.get("stun", {}).get(outcome)
        if value is not None:
            return int(value)
        if outcome == "block" and move.get("block_advantage") is not None:
            remaining = self._action_total(candidate.move_id) - candidate.action_frame
            return max(1, remaining + int(move["block_advantage"]) + 1)
        return 0

    def _prepare_block_spacing(
        self, state: CombatState, actor: int, move_id: str
    ) -> float | None:
        move = self.data.move(move_id)
        spacing = move.get("spacing_on_block")
        if not isinstance(spacing, dict):
            return None
        fighter = state.players[actor]
        target, _ = self.spacing.spacing_after_block(
            move_id, state.distance, fighter.action_recovery_mode
        )
        return target

    def _action_total(self, move_id: str) -> int:
        move = self.data.move(move_id)
        if "total" in move:
            return int(move["total"])
        active = self.calc.active_frames(move_id)
        if active and "recovery" in move:
            return max(active) + int(move["recovery"])
        if active and "landing_recovery" in move:
            return max(active) + int(move["landing_recovery"])
        if "actionable_on_quick_dash_frame" in move:
            return 1
        return max(1, int(move.get("startup", 1)) + int(move.get("recovery", 0)))

    def _is_actionable(self, fighter: FighterState) -> bool:
        return (
            fighter.action_id is None
            and fighter.queued_action is None
            and fighter.stun_remaining == 0
            and fighter.special_stun_remaining == 0
            and not fighter.knockdown
        )

    def _clear_action(self, fighter: FighterState) -> None:
        fighter.action_id = None
        fighter.action_frame = 0
        fighter.action_source = None
        fighter.action_start_distance = 0.0
        fighter.action_recovery_mode = "hold2"
        fighter.action_connected = False
        fighter.contact_action_frame = None
        fighter.contact_outcome = None
        fighter.throw_attempted = False
        fighter.armor_hits_remaining = 0
        fighter.armor_absorbed_hit = False
        fighter.pending_full_spacing = None
        fighter.airborne = False
        fighter.sacrifice_phase = None
        fighter.sacrifice_escape_until = None
        fighter.sacrifice_landing_remaining = 0


def resolve_frame(
    state: CombatState,
    inputs: Sequence[FrameInput] | None = None,
    *,
    resolver: FrameResolver | None = None,
) -> tuple[CombatState, list[FrameEvent]]:
    """Resolve one frame with a reusable resolver or default frame data."""

    return (resolver or FrameResolver()).resolve_frame(state, inputs)

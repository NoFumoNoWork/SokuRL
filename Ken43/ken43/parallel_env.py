from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .env import KenOkiMicrogame
from .diagnostics import EpisodeDiagnostics
from .resolver import CombatState, FrameEvent, FrameInput


MOVEMENT_ACTIONS = (
    "forward_walk", "backwalk", "forward_dash", "backdash", "forward_jump", "back_jump",
)
JINRAI_BRANCHES = ("kazekama", "gorai", "senka")
TRAINING_ROOT_ACTIONS = (
    "2HK", "2HP", "2LK", "2LP", "2MK", "2MP", "5HK", "5HP", "5LK", "5LP", "5MK", "5MP",
    "jHP", "throw_back", "throw_forward", "jinrai_H", "jinrai_L", "jinrai_M",
    "shoryuken_L", "shoryuken_OD", "drive_impact",
)


@dataclass
class PendingSchedule:
    action: str
    remaining: int
    target_delay: int
    actual_delay: int
    submitted_frame: int


@dataclass
class PendingBranch:
    action: str
    delay: int
    remaining: int | None = None
    submitted_frame: int = 0
    window_start_frame: int | None = None


@dataclass
class MovementState:
    kind: str
    frame: int
    total: int


@dataclass(frozen=True)
class ActionChoice:
    label: str
    mode: int
    command: str
    delay: int = 0


class Ken43ParallelEnv(ParallelEnv):
    """Drive-only, frame-stepped corner interaction for two policies."""

    metadata = {"name": "ken43_corner_drive_v2", "is_parallelizable": True, "render_modes": []}
    possible_agents = ["player_0", "player_1"]

    def __init__(
        self,
        *,
        max_frames: int = 600,
        timeout_penalty: float = 0.5,
        time_pressure_start: int = 360,
        time_pressure_total: float = 0.5,
        time_pressure_power: float = 2.0,
        drive_reward_weight: float = 1.0,
        drive_regen_per_frame: float = 0.0025,
        drive_regen_delay: int = 90,
        forward_walk_regen_after: int = 11,
        forward_walk_regen_multiplier: float = 1.5,
        initial_advantages: tuple[int, ...] = (43,),
        schedule_max_delay: int = 10,
        exact_combat_data: bool = True,
        diagnostic_trace_frames: int = 120,
    ):
        self.core = KenOkiMicrogame()
        self.resolver = self.core.resolver
        self.max_frames = int(max_frames)
        self.timeout_penalty = float(timeout_penalty)
        if self.timeout_penalty < 0.0:
            raise ValueError("timeout_penalty must be non-negative")
        self.time_pressure_start = int(time_pressure_start)
        self.time_pressure_total = float(time_pressure_total)
        self.time_pressure_power = float(time_pressure_power)
        if self.time_pressure_start < 0:
            raise ValueError("time_pressure_start must be non-negative")
        if self.time_pressure_total < 0.0:
            raise ValueError("time_pressure_total must be non-negative")
        if self.time_pressure_power <= 0.0:
            raise ValueError("time_pressure_power must be positive")
        pressure_frames = max(0, self.max_frames - self.time_pressure_start)
        self._time_pressure_normalizer = sum(
            (frame / pressure_frames) ** self.time_pressure_power
            for frame in range(1, pressure_frames + 1)
        ) if pressure_frames else 0.0
        self.drive_reward_weight = float(drive_reward_weight)
        self.drive_regen_per_frame = float(drive_regen_per_frame)
        self.drive_regen_delay = int(drive_regen_delay)
        self.forward_walk_regen_after = int(forward_walk_regen_after)
        self.forward_walk_regen_multiplier = float(forward_walk_regen_multiplier)
        self.initial_advantages = tuple(int(value) for value in initial_advantages)
        if not self.initial_advantages or not set(self.initial_advantages) <= {26, 38, 43}:
            raise ValueError("initial_advantages must be a non-empty subset of {26, 38, 43}")
        self.schedule_max_delay = int(schedule_max_delay)
        if not 1 <= self.schedule_max_delay <= 10:
            raise ValueError("schedule_max_delay must be in 1..10")
        self.exact_combat_data = bool(exact_combat_data)
        self.diagnostics = EpisodeDiagnostics(diagnostic_trace_frames)
        self.agents = list(self.possible_agents)
        self._rng = np.random.default_rng()
        self._action_ids = [None, *sorted(self.core.data.moves), *MOVEMENT_ACTIONS]
        self._action_codes = {action: index for index, action in enumerate(self._action_ids)}
        self._root_actions = tuple(
            action for action in TRAINING_ROOT_ACTIONS if action in self.resolver.direct_actions
        ) + MOVEMENT_ACTIONS
        self.unavailable_requested_actions = {}
        self._cancel_targets = tuple(
            action for action in (*TRAINING_ROOT_ACTIONS, *JINRAI_BRANCHES)
            if action in self.resolver.enabled_actions
        )
        self._base_commands = self._build_base_commands()
        self._move_index = {
            command: index for index, command in enumerate(self._base_commands)
        }
        self._action_choices, self.command_index = self._build_action_catalog()
        self.commands = tuple(choice.label for choice in self._action_choices)
        self._action_space = spaces.Discrete(len(self._action_choices))
        self._reset_runtime()
        vector_size = len(self._state_vector(self.core.combat_state, 0))
        self._observation_space = spaces.Dict({
            "observation": spaces.Box(0.0, 10.0, shape=(vector_size,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(len(self._action_choices)),
        })

    def _reset_runtime(self) -> None:
        self._pending: list[PendingSchedule | None] = [None, None]
        self._pending_branch: list[PendingBranch | None] = [None, None]
        self._movements: list[MovementState | None] = [None, None]
        self._regen_cooldown = [0, 0]
        self._forward_walk_streak = [0, 0]
        self._defender_corner_offset = 0.0
        self._movement_terminal: str | None = None
        self._disengaged = False
        self._disengage_cause: str | None = None
        self.initial_advantage = 43

    def _build_base_commands(self) -> tuple[str, ...]:
        commands = ["noop", "guard:standing", "guard:crouching"]
        commands.extend(self._root_actions)
        commands.extend(f"cancel:{action}" for action in self._cancel_targets)
        commands.extend(["branch:no_followup", *(f"branch:{action}" for action in JINRAI_BRANCHES)])
        return tuple(commands)

    def _build_action_catalog(self) -> tuple[tuple[ActionChoice, ...], dict[str, int]]:
        choices: list[ActionChoice] = []
        index: dict[str, int] = {}

        def add(choice: ActionChoice, *aliases: str) -> None:
            action_id = len(choices)
            choices.append(choice)
            index[choice.label] = action_id
            for alias in aliases:
                index[alias] = action_id

        for command in ("noop", "guard:standing", "guard:crouching"):
            add(ActionChoice(command, 0, command))
        for command in self._root_actions:
            add(ActionChoice(f"immediate:{command}", 0, command), command)
            for delay in range(1, self.schedule_max_delay + 1):
                add(ActionChoice(f"schedule:{delay}:{command}", 1, command, delay))
        for command in self._cancel_targets:
            label = f"cancel:{command}"
            add(ActionChoice(label, 0, label))
        for child in ("no_followup", *JINRAI_BRANCHES):
            base = f"branch:{child}"
            for delay in range(11):
                label = f"branch:{delay}:{child}"
                aliases = (base,) if delay == 0 else ()
                add(ActionChoice(label, 0, base, delay), *aliases)
        return tuple(choices), index

    def observation_space(self, agent: str):
        return self._observation_space

    def action_space(self, agent: str):
        return self._action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._reset_runtime()
        requested = options.get("initial_advantage")
        self.initial_advantage = int(
            requested if requested is not None else self._rng.choice(self.initial_advantages)
        )
        self.core.reset(
            knockdown_advantage=self.initial_advantage,
            distance=float(options.get("distance", 70.0)),
            attacker_hp=10000,
            defender_hp=10000,
            attacker_drive=float(options.get("player_0_drive", 6.0)),
            defender_drive=float(options.get("player_1_drive", 6.0)),
        )
        self._disengaged = self.core.combat_state.distance > self._range("2MK", "standing")
        self._disengage_cause = "initial_state" if self._disengaged else None
        self.diagnostics.reset(self.initial_advantage, self.core.combat_state.distance)
        observations = self._observations()
        infos = {agent: {
            "action_labels": self.commands,
            "initial_advantage": self.initial_advantage,
            "unavailable_requested_actions": dict(self.unavailable_requested_actions),
        } for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            raise RuntimeError("step() called after episode termination; call reset()")
        before_distance = self.core.combat_state.distance
        before_corner_offset = self._defender_corner_offset
        before_player_state = [
            {
                "action": fighter.action_id,
                "action_frame": fighter.action_frame,
                "posture": fighter.posture,
                "airborne": fighter.airborne,
                "blocking": fighter.blocking,
                "drive": fighter.drive,
            }
            for fighter in self.core.combat_state.players
        ]
        self._advance_movements()
        before_drive = [fighter.drive for fighter in self.core.combat_state.players]
        frame_inputs: list[FrameInput] = []
        labels: list[str] = []
        for actor, agent in enumerate(self.possible_agents):
            automatic = self._automatic_action(actor)
            if automatic is not None:
                if automatic.startswith("branch:"):
                    frame_inputs.append(FrameInput(
                        branch_to=automatic.split(":", 1)[1],
                        branch_window_override=True,
                    ))
                else:
                    frame_inputs.append(self._input_for_action(actor, automatic))
                labels.append(f"automatic:{automatic}")
                continue
            raw_action = actions.get(agent, self.command_index["noop"])
            try:
                action_id = int(np.asarray(raw_action).item())
            except (TypeError, ValueError):
                action_id = self.command_index["noop"]
            if not self._action_available(actor, action_id):
                action_id = self.command_index["noop"]
            choice = self._action_choices[action_id]
            frame_inputs.append(
                self._decode(actor, choice.mode, choice.command, choice.delay)
            )
            labels.append(choice.label)

        state, events = self.core.resolve_frame(frame_inputs[0], frame_inputs[1])
        for fighter in state.players:
            fighter.hp = 10000
            fighter.burnout = False
            fighter.burnout_pending_action = None
        state.terminal_reason = None
        self._record_motion_events(state, events)
        resolved_drive = [fighter.drive for fighter in state.players]
        bonuses = self._terminal_and_event_bonuses(
            state, events, labels, before_distance
        )
        self._update_regen_cooldowns(events, before_drive, resolved_drive)
        self._apply_drive_regen(state, labels)
        final_drive = [fighter.drive for fighter in state.players]
        drive_changes = [
            self._drive_change_reward(before_drive[i], final_drive[i])
            for i in range(2)
        ]
        multipliers = self._drive_reward_multipliers(events)
        weighted = [drive_changes[i] * multipliers[i] for i in range(2)]
        drive_rewards = [
            self.drive_reward_weight * (weighted[1] - weighted[0]),
            self.drive_reward_weight * (weighted[0] - weighted[1]),
        ]
        terminal = state.terminal_reason is not None
        truncated = state.frame >= self.max_frames and not terminal
        time_pressure = self._time_pressure_reward(state.frame)
        time_penalties = [time_pressure, time_pressure]
        timeout_penalties = (
            [-self.timeout_penalty, -self.timeout_penalty]
            if truncated else [0.0, 0.0]
        )
        rewards = {
            "player_0": drive_rewards[0] + bonuses[0] + time_penalties[0] + timeout_penalties[0],
            "player_1": drive_rewards[1] + bonuses[1] + time_penalties[1] + timeout_penalties[1],
        }
        terminations = {agent: terminal for agent in self.possible_agents}
        truncations = {agent: truncated for agent in self.possible_agents}
        payload = [asdict(event) for event in events]
        step_diagnostics = self.diagnostics.record_step(
            state=state,
            frame_data=self.core.data,
            labels=labels,
            events=events,
            before_drive=before_drive,
            resolved_drive=resolved_drive,
            final_drive=final_drive,
            before_distance=before_distance,
            before_corner_offset=before_corner_offset,
            before_player_state=before_player_state,
            corner_offset=self._defender_corner_offset,
            disengaged=self._disengaged,
            disengage_cause=self._disengage_cause,
            movements=self._movements,
            pending=self._pending,
            pending_branch=self._pending_branch,
            reward_drive=drive_rewards,
            reward_bonus=[
                bonuses[i] + time_penalties[i] + timeout_penalties[i]
                for i in range(2)
            ],
            rewards=[rewards["player_0"], rewards["player_1"]],
        )
        terminal_reason = state.terminal_reason or ("max_frames" if truncated else None)
        episode_diagnostics = (
            self.diagnostics.summary(
                terminal_reason,
                state,
                self._movements,
                self._pending,
                self._pending_branch,
                self._disengaged,
                self._disengage_cause,
            )
            if terminal_reason is not None else None
        )
        infos = {agent: {
            "events": payload,
            "diagnostic_events": step_diagnostics,
            "episode_diagnostics": episode_diagnostics,
            "terminal_reason": terminal_reason,
            "disengaged": self._disengaged,
            "disengage_cause": self._disengage_cause,
            "submitted_command": labels[actor],
            "reward_components": {
                "opponent_drive_change_reward": float(weighted[1 - actor]),
                "own_drive_change_reward": float(weighted[actor]),
                "drive": float(self.drive_reward_weight * (weighted[1 - actor] - weighted[actor])),
                "event_bonus": float(bonuses[actor]),
                "time_pressure": float(time_penalties[actor]),
                "timeout_penalty": float(timeout_penalties[actor]),
            },
        } for actor, agent in enumerate(self.possible_agents)}
        observations = self._observations()
        if terminal or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    @staticmethod
    def _drive_change_reward(before: float, after: float) -> float:
        """Positive for Drive loss and negative for Drive regeneration."""
        return float(before) - float(after)

    def _time_pressure_reward(self, frame: int) -> float:
        if (
            self._time_pressure_normalizer <= 0.0
            or frame <= self.time_pressure_start
        ):
            return 0.0
        pressure_frames = self.max_frames - self.time_pressure_start
        offset = min(frame - self.time_pressure_start, pressure_frames)
        weight = (offset / pressure_frames) ** self.time_pressure_power
        return -self.time_pressure_total * weight / self._time_pressure_normalizer

    def _drive_reward_multipliers(self, events: list[FrameEvent]) -> list[float]:
        result = [1.0, 1.0]
        for event in events:
            if event.type not in {"Hit", "Throw"} or event.target is None:
                continue
            context = event.data.get("counter_context")
            if context == "counter_hit":
                result[event.target] = max(result[event.target], 1.1)
            elif context == "punish_counter":
                result[event.target] = max(result[event.target], 1.2)
        return result

    @staticmethod
    def _award(bonuses: list[float], actor: int, amount: float) -> None:
        bonuses[actor] += amount
        bonuses[1 - actor] -= amount

    def _terminal_and_event_bonuses(
        self,
        state: CombatState,
        events: list[FrameEvent],
        labels: list[str],
        before_distance: float,
    ) -> list[float]:
        bonuses = [0.0, 0.0]
        reason: str | None = None
        disengage_range = self._range("2MK", "standing")
        was_disengaged = self._disengaged
        self._disengaged = state.distance > disengage_range
        if not self._disengaged:
            self._disengage_cause = None
        elif not was_disengaged:
            defender_hit = next(
                (event for event in events if event.type == "Hit" and event.actor == 1),
                None,
            )
            attacker_contact = next(
                (
                    event for event in events
                    if event.type in {"Hit", "Block"} and event.actor == 0
                ),
                None,
            )
            if defender_hit is not None:
                self._disengage_cause = f"defender_hit:{defender_hit.data.get('action')}"
            elif attacker_contact is not None:
                self._disengage_cause = "attacker_contact_pushback"
            elif state.distance > before_distance:
                self._disengage_cause = "movement_or_spacing"
            else:
                self._disengage_cause = "unknown"
        for event in events:
            if event.type == "ThrowClash":
                reason = "throw_tech"
                self._award(bonuses, 1, 0.20 * self.drive_reward_weight)
            elif event.type == "Throw" and event.target == 1:
                reason = "defender_thrown"
                self._award(bonuses, 0, 0.50 * self.drive_reward_weight)
            elif event.type == "Throw" and event.actor == 1 and event.data.get("action") == "throw_back":
                reason = "side_switch"
                self._award(bonuses, 1, 1.00 * self.drive_reward_weight)
            if event.type == "Hit" and event.actor == 1 and event.data.get("action") == "shoryuken_OD":
                reason = "pressure_escape"
                self._award(bonuses, 1, 2.0)
            elif (
                event.type == "Hit"
                and event.actor == 1
                and event.data.get("action") != "drive_impact"
                and self._disengaged
            ):
                reason = "defender_disengage"
                context = event.data.get("counter_context")
                if context == "normal":
                    self._award(bonuses, 1, 0.15 * self.drive_reward_weight)
            if event.type == "Hit" and event.data.get("counter_context") == "punish_counter":
                self._award(bonuses, int(event.actor), 0.50 * self.drive_reward_weight)
            if event.type == "Hit" and event.actor == 1 and event.data.get("counter_context") == "counter_hit":
                if event.data.get("action") not in {"drive_impact", "shoryuken_OD"}:
                    self._award(bonuses, 1, 0.15 * self.drive_reward_weight)
            if event.type == "CounterDI" and event.actor is not None:
                self._award(bonuses, int(event.actor), 0.75 * self.drive_reward_weight)
                if event.actor == 1:
                    reason = "counter_di_escape"
        if (
            state.players[1].special_stun == "wall_splat"
            and any(e.type in {"Hit", "Block"} and e.actor == 0 and e.data.get("action") == "drive_impact" for e in events)
        ):
            reason = "di_wall_splat"
            self._award(bonuses, 0, 0.10 * self.drive_reward_weight)
        empty = [i for i, fighter in enumerate(state.players) if fighter.drive <= 0.0]
        if empty:
            reason = "double_drive_break" if len(empty) == 2 else f"player_{empty[0]}_drive_break"
            if len(empty) == 1:
                self._award(bonuses, 1 - empty[0], 1.00 * self.drive_reward_weight)
        elif reason is None and self._movement_terminal is not None:
            reason = self._movement_terminal
            self._award(bonuses, 1, 1.00 * self.drive_reward_weight)
        elif reason is None and self._defender_corner_offset > 0.0 and state.distance >= self._range("5LP", "standing"):
            reason = "corner_escape"
            self._award(bonuses, 1, 0.75 * self.drive_reward_weight)
        state.terminal_reason = reason
        self._movement_terminal = None
        return bonuses

    def _update_regen_cooldowns(self, events, before_drive, after_drive) -> None:
        for event in events:
            if event.type in {"Hit", "Block", "Throw", "ThrowClash", "ArmorAbsorb"}:
                for index in (event.actor, event.target):
                    if index is not None:
                        self._regen_cooldown[index] = self.drive_regen_delay
        for index in range(2):
            if after_drive[index] < before_drive[index]:
                self._regen_cooldown[index] = self.drive_regen_delay

    def _apply_drive_regen(self, state: CombatState, labels: list[str]) -> None:
        for index, fighter in enumerate(state.players):
            label = labels[index]
            forward_walk = label.endswith("forward_walk")
            self._forward_walk_streak[index] = (
                self._forward_walk_streak[index] + 1 if forward_walk else 0
            )
            if self._regen_cooldown[index] > 0:
                self._regen_cooldown[index] -= 1
                continue
            regen_allowed = (
                fighter.action_id is None
                and (
                    label in {"noop", "guard:standing", "guard:crouching"}
                    or label.endswith("backwalk")
                    or forward_walk
                )
            )
            if regen_allowed and fighter.drive > 0.0:
                multiplier = (
                    self.forward_walk_regen_multiplier
                    if forward_walk
                    and self._forward_walk_streak[index] >= self.forward_walk_regen_after
                    else 1.0
                )
                fighter.drive = min(
                    6.0,
                    fighter.drive + self.drive_regen_per_frame * multiplier,
                )

    def _automatic_action(self, actor: int) -> str | None:
        pending = self._pending[actor]
        if pending is not None:
            if pending.remaining <= 0:
                self._pending[actor] = None
                self.diagnostics.timing_event(
                    "schedule_trigger",
                    self.core.combat_state.frame + 1,
                    actor,
                    action=pending.action,
                    target_delay=pending.target_delay,
                    actual_delay=pending.actual_delay,
                    submitted_frame=pending.submitted_frame,
                )
                return pending.action
            pending.remaining -= 1
        claim = self._pending_branch[actor]
        if claim is not None:
            fighter = self.core.combat_state.players[actor]
            if fighter.action_id is None or not self.core.data.move(fighter.action_id).get("branches"):
                self._pending_branch[actor] = None
                return None
            next_frame = fighter.action_frame + 1
            legal = self.resolver._legal_branch_child(fighter.action_id, next_frame, claim.action)
            if claim.remaining is None and legal is not None:
                claim.remaining = claim.delay
                claim.window_start_frame = self.core.combat_state.frame + 1
                self.diagnostics.timing_event(
                    "jinrai_branch_window",
                    claim.window_start_frame,
                    actor,
                    branch=claim.action,
                    claimed_delay=claim.delay,
                    submitted_frame=claim.submitted_frame,
                )
            if claim.remaining is not None:
                if claim.remaining <= 0:
                    self._pending_branch[actor] = None
                    self.diagnostics.timing_event(
                        "jinrai_branch_trigger",
                        self.core.combat_state.frame + 1,
                        actor,
                        branch=claim.action,
                        claimed_delay=claim.delay,
                        actual_delay=(self.core.combat_state.frame + 1) - int(claim.window_start_frame),
                        window_start_frame=claim.window_start_frame,
                    )
                    return f"branch:{claim.action}"
                claim.remaining -= 1
        return None

    def _decode(self, actor: int, mode: int, command: str, delay: int) -> FrameInput:
        if command == "noop":
            return FrameInput()
        if command.startswith("guard:"):
            return FrameInput(guard=command.split(":", 1)[1])
        if command in self._root_actions and mode == 0:
            return self._input_for_action(actor, command)
        if command in self._root_actions and mode == 1:
            target = delay
            actual = max(0, target + int(self._rng.integers(-2, 3)))
            submitted_frame = self.core.combat_state.frame + 1
            self.diagnostics.timing_event(
                "schedule_claim",
                submitted_frame,
                actor,
                action=command,
                target_delay=target,
                noise=actual - target,
                actual_delay=actual,
            )
            if actual == 0:
                self.diagnostics.timing_event(
                    "schedule_trigger",
                    submitted_frame,
                    actor,
                    action=command,
                    target_delay=target,
                    actual_delay=0,
                    submitted_frame=submitted_frame,
                )
                return self._input_for_action(actor, command)
            self._pending[actor] = PendingSchedule(
                command, actual - 1, target, actual, submitted_frame
            )
            return FrameInput()
        if command.startswith("cancel:"):
            return FrameInput(cancel_to=command.split(":", 1)[1])
        if command.startswith("branch:"):
            child = command.split(":", 1)[1]
            if child == "no_followup":
                self._pending_branch[actor] = None
            else:
                submitted_frame = self.core.combat_state.frame + 1
                self._pending_branch[actor] = PendingBranch(
                    child, delay, submitted_frame=submitted_frame
                )
                self.diagnostics.timing_event(
                    "jinrai_branch_claim",
                    submitted_frame,
                    actor,
                    branch=child,
                    claimed_delay=delay,
                )
            return FrameInput()
        raise ValueError(f"unknown command: {command}")

    @staticmethod
    def _format_action(mode: int, command: str, delay: int) -> str:
        if command.startswith("branch:"):
            return f"{command}(delay={delay})"
        return f"schedule:{delay}:{command}" if mode == 1 else f"immediate:{command}"

    def _input_for_action(self, actor: int, action: str) -> FrameInput:
        if action in {"forward_walk", "backwalk"}:
            self._walk(actor, action)
            return FrameInput()
        if action in MOVEMENT_ACTIONS:
            total = {"forward_dash": 19, "backdash": 23, "forward_jump": 46, "back_jump": 46}[action]
            self._movements[actor] = MovementState(action, 1, total)
            return FrameInput()
        return FrameInput(action=action)

    def _walk(self, actor: int, action: str) -> None:
        state = self.core.combat_state
        amount = 3.2
        if action == "forward_walk":
            if actor == 1 and state.distance <= amount:
                self._movement_terminal = "side_switch"
            state.distance = max(0.0, state.distance - amount)
            if actor == 1:
                self._defender_corner_offset += amount
        elif actor == 0:
            state.distance += amount
        elif self._defender_corner_offset > 0.0:
            moved = min(amount, self._defender_corner_offset)
            self._defender_corner_offset -= moved
            state.distance += moved

    def _advance_movements(self) -> None:
        state = self.core.combat_state
        for actor, motion in enumerate(self._movements):
            if motion is None:
                continue
            fighter = state.players[actor]
            if (
                fighter.stun_remaining > 0
                or fighter.knockdown
                or fighter.special_stun is not None
            ):
                fighter.airborne = False
                self._movements[actor] = None
                continue
            motion.frame += 1
            if "jump" in motion.kind:
                fighter.airborne = motion.frame >= 5
            if motion.frame < motion.total:
                continue
            before = state.distance
            displacement = {"backdash": 93.10, "forward_dash": 133.40, "forward_jump": 110.0, "back_jump": 152.0}[motion.kind]
            if motion.kind.startswith("forward"):
                if actor == 1:
                    self._defender_corner_offset += min(before, displacement)
                    if motion.kind == "forward_jump" and displacement > before:
                        self._movement_terminal = "side_switch"
                state.distance = abs(before - displacement) if motion.kind == "forward_jump" else max(0.0, before - displacement)
            elif actor == 0:
                state.distance = before + displacement
            elif self._defender_corner_offset > 0.0:
                amount = min(displacement, self._defender_corner_offset)
                self._defender_corner_offset -= amount
                state.distance = before + amount
            fighter.airborne = False
            self._movements[actor] = None

    def _record_motion_events(self, state: CombatState, events: list[FrameEvent]) -> None:
        for event in list(events):
            if event.type not in {"Hit", "Throw"} or event.target is None:
                continue
            target = int(event.target)
            motion = self._movements[target]
            if motion is None:
                continue
            self._movements[target] = None
            state.players[target].airborne = False
            events.append(FrameEvent(
                "MovementCancelled",
                target,
                None,
                {
                    "movement": motion.kind,
                    "movement_frame": motion.frame,
                    "cause": event.type.lower(),
                },
            ))

    def _range(self, move_id: str, posture: str) -> float:
        return float(self.core.spacing.effective_range(move_id, posture))

    def _base_command_available(self, state: CombatState, actor: int, action: str) -> bool:
        fighter = state.players[actor]
        if action in MOVEMENT_ACTIONS:
            return self.resolver._is_actionable(fighter) and self._movements[actor] is None
        if action == "jHP":
            return bool(fighter.airborne and self.resolver._is_actionable(fighter))
        return self.resolver.direct_action_available(state, actor, action, exact_combat=self.exact_combat_data)

    def _state_for_mask(self, actor: int) -> CombatState:
        state = self.core.combat_state
        if actor == 1 and state.players[1].knockdown and state.frame + 1 >= self.core.knockdown_advantage:
            state = deepcopy(state)
            state.players[1].knockdown = False
            state.players[1].throw_invul_remaining = 1
        return state

    def _move_mask(self, actor: int) -> np.ndarray:
        state = self._state_for_mask(actor)
        fighter = state.players[actor]
        mask = np.zeros(len(self._base_commands), dtype=np.int8)
        for command in ("noop", "guard:standing", "guard:crouching"):
            mask[self._move_index[command]] = 1
        busy = (
            self._pending[actor] is not None
            or self._movements[actor] is not None
        )
        if not busy:
            for action in self._root_actions:
                available = self._base_command_available(state, actor, action)
                mask[self._move_index[action]] = int(available)
        elif (
            fighter.airborne
            and self._movements[actor] is not None
            and "jump" in self._movements[actor].kind
            and self._pending[actor] is None
        ):
            mask[self._move_index["jHP"]] = int(
                self._base_command_available(state, actor, "jHP")
            )
        if fighter.action_id is not None:
            next_frame = fighter.action_frame + 1
            for action in self._cancel_targets:
                prior_contact = fighter.contact_action_frame is not None
                active_contact = self.resolver._active_index(fighter.action_id, next_frame) is not None
                delay = next_frame - int(fighter.contact_action_frame) if prior_contact else 0
                legal = (prior_contact or active_contact) and self.resolver._cancel_allowed(fighter.action_id, action, delay, next_frame)
                mask[self._move_index[f"cancel:{action}"]] = int(legal)
            branch_parent = bool(
                self.core.data.move(fighter.action_id).get("branches")
                and self._pending_branch[actor] is None
            )
            mask[self._move_index["branch:no_followup"]] = int(branch_parent)
            for child in JINRAI_BRANCHES:
                mask[self._move_index[f"branch:{child}"]] = int(branch_parent)
        return mask

    def _action_available(self, actor: int, action_id: int) -> bool:
        if not 0 <= action_id < len(self._action_choices):
            return False
        choice = self._action_choices[action_id]
        return bool(self._move_mask(actor)[self._move_index[choice.command]])

    def _mask(self, actor: int) -> np.ndarray:
        move_mask = self._move_mask(actor)
        return np.asarray(
            [move_mask[self._move_index[choice.command]] for choice in self._action_choices],
            dtype=np.int8,
        )

    def _observations(self) -> dict[str, dict[str, np.ndarray]]:
        return {agent: {
            "observation": self._state_vector(self.core.combat_state, actor),
            "action_mask": self._mask(actor),
        } for actor, agent in enumerate(self.possible_agents)}

    def _state_vector(self, state: CombatState, actor: int) -> np.ndarray:
        own, opponent = state.players[actor], state.players[1 - actor]
        pending = self._pending[actor]
        special = {None: 0.0, "wall_splat": 0.5, "crumple": 1.0}
        own_action = own.action_id
        own_action_frame = own.action_frame
        if own_action is None and self._movements[actor] is not None:
            own_action = self._movements[actor].kind
            own_action_frame = self._movements[actor].frame
        values = [
            state.frame / max(1, self.max_frames), state.distance / 400.0,
            self.initial_advantage / 43.0, self._defender_corner_offset / 400.0,
            float(self._disengaged),
            float(pending is not None),
            pending.remaining / max(1, self.schedule_max_delay) if pending else 0.0,
            float(self._pending_branch[actor] is not None),
            own.drive / 6.0, float(own.posture == "crouching"),
            self._action_codes.get(own_action, 0) / max(1, len(self._action_ids) - 1),
            own_action_frame / 100.0, own.stun_remaining / 100.0,
            own.throw_invul_remaining / 10.0, float(own.knockdown), float(own.airborne),
            float(own.blocking), own.armor_hits_remaining / 2.0,
            special.get(own.special_stun, 0.0), own.special_stun_remaining / 150.0,
            self._regen_cooldown[actor] / max(1, self.drive_regen_delay),
            opponent.drive / 6.0, float(opponent.posture == "crouching"),
            opponent.stun_remaining / 100.0, opponent.throw_invul_remaining / 10.0,
            float(opponent.knockdown), float(opponent.airborne), float(opponent.blocking),
            opponent.armor_hits_remaining / 2.0, special.get(opponent.special_stun, 0.0),
            opponent.special_stun_remaining / 150.0,
            self._regen_cooldown[1 - actor] / max(1, self.drive_regen_delay),
        ]
        return np.asarray(values, dtype=np.float32)

    def state(self) -> np.ndarray:
        return np.concatenate([self._state_vector(self.core.combat_state, 0), self._state_vector(self.core.combat_state, 1)])

    def render(self):
        return None

    def close(self):
        return None

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


GROUND_NORMALS = {
    "5LP", "2LP", "5LK", "2LK", "5MP", "2MP",
    "5MK", "2MK", "5HP", "2HP", "5HK", "2HK",
}


class PressureState(str, Enum):
    TRUE_BLOCKSTRING = "true_blockstring"
    ZERO_GAP = "zero_gap"
    CHALLENGEABLE_GAP = "challengeable_gap"
    NEGATIVE_BUT_CANCELABLE = "negative_but_cancelable"
    TURN_ENDED = "turn_ended"
    PLUS_PRESSURE = "plus_pressure"


@dataclass(frozen=True)
class BlockAssessment:
    state: PressureState
    advantage: int | None
    cancel_open: bool
    gap: int | None = None


@dataclass
class TacticalMemory:
    last_block: dict[str, Any] | None = None
    continuation_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    challenge_attempt: tuple[str, str] | None = None
    pending_challenge: str | None = None
    challenge_delay: int = 0


def _block_advantage(action: str, active_index: int, env) -> int | None:
    try:
        move = env.core.data.move(action)
        if move.get("block_advantage") is not None:
            return int(move["block_advantage"]) + max(0, active_index - 1)
        return int(env.core.calc.raw_advantage(action, "block", active_index))
    except (KeyError, TypeError, ValueError):
        return None


def _cancel_open(move: dict[str, Any]) -> bool:
    cancel = move.get("cancel", {})
    for channel in ("chain", "special"):
        if bool(cancel.get(channel, {}).get("enabled")):
            return True
    return bool(cancel.get("target_combo", {}).get("targets"))


def classify_pressure_state(
    env,
    action: str,
    active_index: int = 1,
    *,
    gap: int | None = None,
    committed_recovery: bool = True,
) -> BlockAssessment:
    move = env.core.data.move(action)
    advantage = _block_advantage(action, active_index, env)
    cancel_open = _cancel_open(move)
    if gap is not None:
        if gap < 0:
            state = PressureState.TRUE_BLOCKSTRING
        elif gap == 0:
            state = PressureState.ZERO_GAP
        else:
            state = PressureState.CHALLENGEABLE_GAP
        return BlockAssessment(state, advantage, cancel_open, gap)
    if advantage is not None and advantage > 0:
        state = PressureState.PLUS_PRESSURE
    elif cancel_open:
        state = PressureState.NEGATIVE_BUT_CANCELABLE
    elif committed_recovery:
        state = PressureState.TURN_ENDED
    else:
        state = PressureState.PLUS_PRESSURE
    return BlockAssessment(state, advantage, cancel_open)


def find_guaranteed_punish(
    env,
    action: str,
    active_index: int,
    distance: float,
    opponent_posture: str,
    *,
    cancel_resolved: bool = False,
) -> tuple[str | None, BlockAssessment]:
    assessment = classify_pressure_state(env, action, active_index)
    if cancel_resolved and assessment.state is PressureState.NEGATIVE_BUT_CANCELABLE:
        assessment = BlockAssessment(
            PressureState.TURN_ENDED,
            assessment.advantage,
            False,
            assessment.gap,
        )
    if (
        assessment.state is not PressureState.TURN_ENDED
        or assessment.advantage is None
        or assessment.advantage > -4
    ):
        return None, assessment

    punish_window = -assessment.advantage
    scored: list[tuple[float, str]] = []
    for move_id in ("5HP", "5MP", "2MP", "2MK", "5LP", "2LP", "throw_forward"):
        try:
            move = env.core.data.move(move_id)
            startup = int(move["startup"])
            reach = float(env._range(move_id, opponent_posture))
        except (KeyError, TypeError, ValueError):
            continue
        if startup > punish_window or distance > reach:
            continue
        damage = float(move.get("damage") or 0.0)
        safety_margin = punish_window - startup
        reach_margin = max(0.0, reach - distance)
        throw_penalty = 1000.0 if move_id.startswith("throw_") else 0.0
        score = damage + 20.0 * safety_margin + min(100.0, reach_margin) - throw_penalty
        scored.append((score, move_id))
    if not scored:
        return None, assessment
    scored.sort(reverse=True)
    return scored[0][1], assessment


def _cancel_window_end_action_frame(move: dict[str, Any], contact_frame: int, env) -> int:
    end = int(contact_frame)
    cancel = move.get("cancel", {})
    special = cancel.get("special", {})
    if special.get("enabled"):
        delays = special.get("default_delay_frames", [0])
        if delays:
            end = max(end, int(contact_frame) + max(int(value) for value in delays))
    chain = cancel.get("chain", {})
    if chain.get("enabled") and chain.get("delayable"):
        try:
            end = max(end, max(env.core.calc.active_frames(move["id"])))
        except (KeyError, TypeError, ValueError):
            pass
    return end


def claim_deferred_punish(env, tactics: TacticalMemory, actor: int) -> str | None:
    context = tactics.last_block
    if context is None or context.get("assessment") != PressureState.NEGATIVE_BUT_CANCELABLE.value:
        return None
    opponent = 1 - actor
    opponent_state = env.core.combat_state.players[opponent]
    action = str(context["action"])
    if opponent_state.action_id != action:
        return None
    move = dict(env.core.data.move(action))
    move["id"] = action
    cancel_end = _cancel_window_end_action_frame(
        move, int(context["contact_action_frame"]), env
    )
    if int(opponent_state.action_frame) <= cancel_end:
        return None
    punish, _ = find_guaranteed_punish(
        env,
        action,
        int(context["active_index"]),
        env.core.combat_state.distance,
        opponent_state.posture,
        cancel_resolved=True,
    )
    if punish is not None:
        tactics.last_block = None
    return punish


def choose_challenge(
    env,
    *,
    gap: int,
    distance: float,
    opponent_posture: str,
    repetition: int,
    rng,
) -> str | None:
    if gap <= 0:
        return None
    if gap < 4:
        if repetition >= 3 and rng.random() < 0.12:
            return "shoryuken_OD"
        return None
    probability = (0.20, 0.50, 0.75, 0.90)[min(max(repetition, 1), 4) - 1]
    if rng.random() >= probability:
        return None
    for move_id in ("5LP", "2LP"):
        try:
            if int(env.core.data.move(move_id)["startup"]) <= gap and distance <= env._range(
                move_id, opponent_posture
            ):
                return move_id
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _actionable_in(fighter, env) -> int:
    if fighter.special_stun_remaining > 0:
        return int(fighter.special_stun_remaining)
    if fighter.stun_remaining > 0:
        return int(fighter.stun_remaining)
    if fighter.knockdown:
        return 10_000
    if fighter.action_id is None:
        return 0
    try:
        total = int(env.resolver._action_total(fighter.action_id))
    except (KeyError, TypeError, ValueError):
        return 10_000
    return max(0, total - int(fighter.action_frame) + 1)


def choose_anti_air(
    env,
    *,
    actor: int,
    jumper: int,
    movement,
    jump_count: int,
    rng,
    recognition_delay: int = 2,
) -> str | None:
    if movement is None or "jump" not in movement.kind:
        return None
    if movement.frame <= recognition_delay:
        return None

    state = env.core.combat_state
    defender = state.players[actor]
    jumping_fighter = state.players[jumper]
    if jumping_fighter.action_id == "jHP" and jumping_fighter.action_frame >= 1:
        startup = int(env.core.data.move("jHP")["startup"])
        contact_eta = max(1, startup - int(jumping_fighter.action_frame) + 1)
    else:
        contact_eta = max(1, 40 - int(movement.frame))
    available = contact_eta - _actionable_in(defender, env)
    if available <= 0:
        return None

    probability = (0.55, 0.70, 0.85, 0.90)[min(max(jump_count, 1), 4) - 1]
    if rng.random() >= probability:
        return None

    distance = float(state.distance)
    posture = jumping_fighter.posture
    candidates: list[tuple[float, str]] = []
    try:
        two_hp_reach = float(env._range("2HP", posture))
    except (KeyError, TypeError, ValueError):
        two_hp_reach = 0.0
    dp_trajectory_valid = movement.kind in {
        "forward_jump", "vertical_jump", "neutral_jump",
    }
    if (
        dp_trajectory_valid
        and int(env.core.data.move("shoryuken_L")["startup"]) <= available
        and distance <= two_hp_reach
    ):
        candidates.append((0.90, "shoryuken_L"))
    if (
        int(env.core.data.move("2HP")["startup"]) <= available
        and distance <= two_hp_reach
    ):
        candidates.append((0.72, "2HP"))
    if (
        not candidates
        and defender.drive >= 2.0
        and int(env.core.data.move("shoryuken_OD")["startup"]) <= available
        and dp_trajectory_valid
        and distance <= two_hp_reach
        and rng.random() < 0.08
    ):
        candidates.append((0.55, "shoryuken_OD"))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _legal_action(observation, command_index, preferred, fallback="noop") -> int:
    mask = observation["action_mask"]
    for command in preferred:
        action = command_index.get(command)
        if action is None:
            continue
        if mask[int(action)]:
            return int(action)
    return int(command_index[fallback])


def select_bot_action(bot, observation, command_index, rng, env=None, actor: int = 1) -> int:
    contextual_act = getattr(bot, "act_with_context", None)
    if env is not None and callable(contextual_act):
        return int(contextual_act(observation, command_index, rng, env, actor))
    return int(bot.act(observation, command_index, rng))


def update_bot(bot, infos, env=None, actor: int = 1) -> None:
    observe = getattr(bot, "observe", None)
    if env is not None and callable(observe):
        observe(infos, env, actor)


@dataclass
class AlwaysBlockBot:
    posture: str = "crouching"

    def act(self, observation, command_index, rng):
        return _legal_action(observation, command_index, [f"guard:{self.posture}"])


class AlwaysThrowBot:
    def act(self, observation, command_index, rng):
        return _legal_action(observation, command_index, ["immediate:throw_forward", "guard:crouching"])


class AlwaysMashBot:
    def act(self, observation, command_index, rng):
        return _legal_action(observation, command_index, ["immediate:5LP", "guard:crouching"])


class AlwaysODDPBot:
    def act(self, observation, command_index, rng):
        return _legal_action(observation, command_index, ["immediate:shoryuken_OD", "guard:crouching"])


class AlwaysDIBot:
    def act(self, observation, command_index, rng):
        return _legal_action(observation, command_index, ["immediate:drive_impact", "guard:crouching"])


class AlwaysJinraiBot:
    def act(self, observation, command_index, rng):
        return _legal_action(
            observation, command_index,
            ["branch:0:kazekama", "immediate:jinrai_M", "guard:crouching"],
        )


@dataclass
class BackdashHeavyBot:
    use_backdash: bool = True

    def act(self, observation, command_index, rng):
        preferred = (
            ["immediate:backdash", "immediate:5HP", "guard:crouching"]
            if self.use_backdash
            else ["immediate:5HP", "immediate:backdash", "guard:crouching"]
        )
        result = _legal_action(observation, command_index, preferred)
        for key in ("immediate:backdash", "immediate:5HP"):
            if command_index[key] == result:
                self.use_backdash = not self.use_backdash
                break
        return result


class RandomBot:
    def act(self, observation, command_index, rng):
        candidates = [
            key for key in command_index
            if key in {"noop", "guard:standing", "guard:crouching"}
            or key.startswith(("immediate:", "schedule:", "cancel:", "branch:"))
        ]
        rng.shuffle(candidates)
        return _legal_action(observation, command_index, candidates)


@dataclass
class HeuristicOffenseBot:
    """Persistent P1 offense macros used to train and evaluate a defender."""

    forced_macro: str | None = None
    persistence: float = 0.60
    macro: str | None = field(init=False, default=None)
    sequence: list[str] = field(init=False, default_factory=list)
    recent_outcomes: deque[str] = field(init=False)
    macro_weights: dict[str, float] = field(init=False)
    last_successful_macro: str | None = field(init=False, default=None)
    jump_attack_frame: int = field(init=False, default=33)
    oki_contact_frame: int | None = field(init=False, default=None)
    reactive_response: str | None = field(init=False, default=None)
    reactive_trigger: str | None = field(init=False, default=None)
    seen_jump_start: int | None = field(init=False, default=None)
    seen_di_start: int | None = field(init=False, default=None)
    opponent_jump_seen_count: int = field(init=False, default=0)
    opponent_di_seen_count: int = field(init=False, default=0)
    seen_poke_start: int | None = field(init=False, default=None)
    opponent_poke_seen_count: int = field(init=False, default=0)
    tactics: TacticalMemory = field(init=False)

    def __post_init__(self) -> None:
        valid = {None, "safe_jump", "strike_throw", "stagger", "commitment"}
        if self.forced_macro not in valid:
            raise ValueError(f"unknown offense macro: {self.forced_macro}")
        self.reset()

    def reset(self) -> None:
        self.macro = None
        self.sequence = []
        self.recent_outcomes = deque(maxlen=5)
        self.macro_weights = {
            "safe_jump": 1.0,
            "strike_throw": 1.0,
            "stagger": 1.0,
            "commitment": 1.0,
        }
        self.last_successful_macro = None
        self.jump_attack_frame = 33
        self.oki_contact_frame = None
        self.reactive_response = None
        self.reactive_trigger = None
        self.seen_jump_start = None
        self.seen_di_start = None
        self.opponent_jump_seen_count = 0
        self.opponent_di_seen_count = 0
        self.seen_poke_start = None
        self.opponent_poke_seen_count = 0
        self.tactics = TacticalMemory()

    def act(self, observation, command_index, rng):
        return _legal_action(observation, command_index, ["guard:crouching"])

    def act_with_context(self, observation, command_index, rng, env, actor: int = 0):
        state = env.core.combat_state
        fighter = state.players[actor]
        defender = state.players[1 - actor]
        movement = env._movements[actor]
        defender_movement = env._movements[1 - actor]

        deferred_punish = claim_deferred_punish(env, self.tactics, actor)
        if deferred_punish is not None:
            self.reactive_response = deferred_punish
            self.reactive_trigger = "guaranteed_punish"
            self.recent_outcomes.append(
                f"guaranteed_punish_deferred:{defender.action_id}"
            )

        if self.tactics.pending_challenge is not None:
            if self.tactics.challenge_delay > 0:
                self.tactics.challenge_delay -= 1
                return _legal_action(
                    observation, command_index, ["guard:crouching"], fallback="noop"
                )
            self.reactive_response = self.tactics.pending_challenge
            self.reactive_trigger = "challenge"
            self.tactics.pending_challenge = None

        self._update_reactive_response(
            env, actor, defender_movement, rng
        )
        if self.reactive_response is not None:
            fallback = (
                "guard:standing"
                if defender_movement is not None and "jump" in defender_movement.kind
                else "guard:crouching"
            )
            command = (
                f"cancel:{self.reactive_response}"
                if fighter.action_id is not None
                else f"immediate:{self.reactive_response}"
            )
            result = _legal_action(
                observation, command_index, [command, fallback], fallback="noop"
            )
            if result == command_index.get(command):
                self.reactive_response = None
                self.reactive_trigger = None
            return result

        if defender_movement is not None and "jump" in defender_movement.kind:
            guard = (
                "guard:crouching"
                if defender_movement.frame <= 2
                else "guard:standing"
            )
            return _legal_action(
                observation, command_index, [guard], fallback="noop"
            )
        if defender.action_id in GROUND_NORMALS:
            return _legal_action(
                observation, command_index, ["guard:crouching"], fallback="noop"
            )

        if not self.sequence and fighter.action_id is None and movement is None:
            self._start_macro(rng, defender.knockdown, env.initial_advantage)

        if not self.sequence:
            return _legal_action(observation, command_index, ["guard:crouching"])

        step = self.sequence[0]
        if defender.knockdown and self.oki_contact_frame is not None and fighter.action_id is None:
            startup = self._startup_for_step(step, env)
            if startup is not None and state.frame < self.oki_contact_frame - startup:
                return _legal_action(observation, command_index, ["noop"])

        if movement is not None:
            if step == "jHP" and "jump" in movement.kind:
                if movement.frame < self.jump_attack_frame:
                    return _legal_action(observation, command_index, ["noop"])
                return self._commit_step(
                    observation, command_index, "immediate:jHP"
                )
            return _legal_action(observation, command_index, ["noop"])

        if fighter.action_id is not None:
            if step.startswith("branch:"):
                return self._commit_step(observation, command_index, step)
            return self._commit_step(
                observation, command_index, f"cancel:{step}", fallback="noop"
            )

        command = step if step.startswith(("guard:", "schedule:")) else f"immediate:{step}"
        return self._commit_step(observation, command_index, command)

    def _update_reactive_response(self, env, actor, defender_movement, rng) -> None:
        state = env.core.combat_state
        defender = state.players[1 - actor]
        if self.reactive_trigger == "jump" and defender_movement is None:
            self.reactive_response = None
            self.reactive_trigger = None
        if self.reactive_trigger == "di" and defender.action_id != "drive_impact":
            self.reactive_response = None
            self.reactive_trigger = None
        if self.reactive_trigger == "poke" and defender.action_id not in GROUND_NORMALS:
            self.reactive_response = None
            self.reactive_trigger = None

        if defender.action_id == "drive_impact":
            start = state.frame - defender.action_frame
            if self.seen_di_start != start:
                self.seen_di_start = start
                self.opponent_di_seen_count += 1
                probability = (0.85, 0.95, 1.0)[
                    min(self.opponent_di_seen_count, 3) - 1
                ]
                if rng.random() < probability:
                    self.reactive_response = "drive_impact"
                    self.reactive_trigger = "di"
                    self.recent_outcomes.append("counter_di_claim")
            return

        if defender_movement is not None and "jump" in defender_movement.kind:
            if defender_movement.frame <= 2:
                return
            start = state.frame - defender_movement.frame
            if self.seen_jump_start != start:
                self.seen_jump_start = start
                self.opponent_jump_seen_count += 1
                response = choose_anti_air(
                    env,
                    actor=actor,
                    jumper=1 - actor,
                    movement=defender_movement,
                    jump_count=self.opponent_jump_seen_count,
                    rng=rng,
                )
                if response is not None:
                    self.reactive_response = response
                    self.reactive_trigger = "jump"
                    self.recent_outcomes.append("anti_jump_claim")
            return

        if defender.action_id in GROUND_NORMALS:
            start = state.frame - defender.action_frame
            if self.seen_poke_start != start:
                self.seen_poke_start = start
                self.opponent_poke_seen_count += 1
                if self.opponent_poke_seen_count >= 2:
                    probability = 0.75 if self.opponent_poke_seen_count == 2 else 1.0
                    if rng.random() < probability:
                        self.reactive_response = "drive_impact"
                        self.reactive_trigger = "poke"
                        self.recent_outcomes.append("anti_poke_claim")

    def _commit_step(self, observation, command_index, command: str, fallback: str = "noop") -> int:
        result = _legal_action(observation, command_index, [command], fallback=fallback)
        if command_index.get(command) == result:
            self.sequence.pop(0)
        return result

    @staticmethod
    def _startup_for_step(step: str, env) -> int | None:
        if step in {"forward_jump", "back_jump"} or step.startswith(("guard:", "branch:")):
            return None
        move_id = step.split(":")[-1]
        try:
            return int(env.core.data.move(move_id)["startup"])
        except (KeyError, TypeError, ValueError):
            return None

    def _start_macro(self, rng, defender_knockdown: bool, initial_advantage: int) -> None:
        if self.forced_macro is not None:
            macro = self.forced_macro
        elif self.last_successful_macro is not None and rng.random() < self.persistence:
            macro = self.last_successful_macro
        else:
            names = np.asarray(list(self.macro_weights))
            weights = np.asarray([self.macro_weights[name] for name in names], dtype=float)
            weights /= weights.sum()
            macro = str(rng.choice(names, p=weights))
        self.macro = macro
        self.oki_contact_frame = initial_advantage if defender_knockdown else None

        if macro == "safe_jump":
            self.jump_attack_frame = int(rng.integers(32, 35))
            finisher = str(rng.choice(["throw_forward", "5HP", "guard:crouching"]))
            self.sequence = ["forward_jump", "jHP", finisher]
            self.oki_contact_frame = None
        elif macro == "strike_throw":
            routes = [
                ("5LP", "throw_forward"),
                ("5LP", "5LP", "throw_forward"),
                ("5MP", "throw_forward"),
            ]
            self.sequence = list(routes[int(rng.integers(0, len(routes)))])
        elif macro == "stagger":
            chains = [
                ("2LK", "5LP", "5LP", "5LK"),
                ("5LP", "2LK", "5LP", "5HP"),
                ("2LP", "2LP", "5LP", "5LK"),
                ("5MP", "2MP", "5HP"),
            ]
            self.sequence = list(chains[int(rng.integers(0, len(chains)))])
        else:
            routes = [
                ("5HP", "jinrai_M", "branch:0:kazekama"),
                ("2MP", "jinrai_M", "branch:0:gorai"),
                ("5LP", "jinrai_L", "branch:0:kazekama"),
                ("2LK", "drive_impact"),
            ]
            self.sequence = list(routes[int(rng.integers(0, len(routes)))])

    def observe(self, infos: dict[str, Any], env, actor: int = 0) -> None:
        opponent = 1 - actor
        own_info = infos[f"player_{actor}"]
        for event in own_info.get("events", []):
            event_type = event.get("type")
            event_actor = event.get("actor")
            target = event.get("target")
            data = event.get("data", {})
            action = str(data.get("action", ""))
            if event_type == "Block" and event_actor == opponent and target == actor:
                active_index = int(data.get("active_index", 1))
                punish, assessment = find_guaranteed_punish(
                    env,
                    action,
                    active_index,
                    env.core.combat_state.distance,
                    env.core.combat_state.players[opponent].posture,
                )
                self.tactics.last_block = {
                    "action": action,
                    "frame": env.core.combat_state.frame,
                    "stun": int(data.get("stun", 0)),
                    "active_index": active_index,
                    "contact_action_frame": int(
                        env.core.combat_state.players[opponent].action_frame
                    ),
                    "assessment": assessment.state.value,
                }
                if punish is not None:
                    self.reactive_response = punish
                    self.reactive_trigger = "guaranteed_punish"
                    self.recent_outcomes.append(f"guaranteed_punish:{action}")
            if (
                event_type in {"ActionStarted", "ReversalStarted"}
                and event_actor == opponent
                and self.tactics.last_block is not None
            ):
                previous = self.tactics.last_block
                try:
                    startup = int(env.core.data.move(action)["startup"])
                except (KeyError, TypeError, ValueError):
                    startup = 10_000
                gap = (
                    env.core.combat_state.frame + startup - 1
                    - (int(previous["frame"]) + int(previous["stun"]))
                )
                key = (str(previous["action"]), action)
                repetition = self.tactics.continuation_counts.get(key, 0) + 1
                self.tactics.continuation_counts[key] = repetition
                challenge = choose_challenge(
                    env,
                    gap=gap - 2,
                    distance=env.core.combat_state.distance,
                    opponent_posture=env.core.combat_state.players[opponent].posture,
                    repetition=repetition,
                    rng=env._rng,
                )
                if challenge is not None:
                    self.tactics.pending_challenge = challenge
                    self.tactics.challenge_delay = 2
                    self.tactics.challenge_attempt = key
                    self.recent_outcomes.append(f"challenge:{gap}:{action}")
                self.tactics.last_block = None
            if self.tactics.challenge_attempt is not None and event_type == "Hit":
                key = self.tactics.challenge_attempt
                if event_actor == actor and target == opponent:
                    self.tactics.continuation_counts[key] = min(
                        6, self.tactics.continuation_counts.get(key, 1) + 1
                    )
                    self.recent_outcomes.append("challenge_success")
                    self.tactics.challenge_attempt = None
                elif event_actor == opponent and target == actor:
                    self.tactics.continuation_counts[key] = max(
                        1, self.tactics.continuation_counts.get(key, 1) - 1
                    )
                    self.recent_outcomes.append("challenge_counter_hit")
                    self.tactics.challenge_attempt = None
            if event_actor == actor and target == opponent and event_type in {"Hit", "Throw"}:
                self.last_successful_macro = self.macro
                if self.macro is not None:
                    self.macro_weights[self.macro] = min(
                        1.5, self.macro_weights[self.macro] * 1.10
                    )
                self.recent_outcomes.append(f"success:{self.macro}")
            elif event_actor == opponent and target == actor and event_type in {"Hit", "Throw"}:
                if self.macro is not None:
                    self.macro_weights[self.macro] = max(
                        0.5, self.macro_weights[self.macro] * 0.90
                    )
                self.recent_outcomes.append(f"countered:{self.macro}")


@dataclass
class HeuristicDefenderBot:
    """Stateful P2 decision tree for the corner-defense training role."""

    memory_size: int = 5
    persistence: float = 0.675
    recent_events: deque[str] = field(init=False)
    biases: dict[str, float] = field(init=False)
    blocked_light_count: int = field(init=False, default=0)
    normal_to_special_seen_count: int = field(init=False, default=0)
    attacker_counter_di_count: int = field(init=False, default=0)
    last_blocked_action: str | None = field(init=False, default=None)
    last_attacker_command: str = field(init=False, default="noop")
    successful_family: str | None = field(init=False, default=None)
    current_family: str | None = field(init=False, default=None)
    pending_response: str | None = field(init=False, default=None)
    approach_family: str | None = field(init=False, default=None)
    wakeup_decided: bool = field(init=False, default=False)
    seen_di_start_frame: int | None = field(init=False, default=None)
    opponent_di_seen_count: int = field(init=False, default=0)
    jhp_hit_count: int = field(init=False, default=0)
    jhp_block_count: int = field(init=False, default=0)
    seen_jump_start: int | None = field(init=False, default=None)
    opponent_jump_seen_count: int = field(init=False, default=0)
    tactics: TacticalMemory = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.recent_events = deque(maxlen=self.memory_size)
        self.biases = {
            "tech": 1.0,
            "mash": 1.0,
            "reversal": 1.0,
            "di": 1.0,
            "jump": 1.0,
            "block": 1.0,
        }
        self.blocked_light_count = 0
        self.normal_to_special_seen_count = 0
        self.attacker_counter_di_count = 0
        self.last_blocked_action = None
        self.last_attacker_command = "noop"
        self.successful_family = None
        self.current_family = None
        self.pending_response = None
        self.approach_family = None
        self.wakeup_decided = False
        self.seen_di_start_frame = None
        self.opponent_di_seen_count = 0
        self.jhp_hit_count = 0
        self.jhp_block_count = 0
        self.seen_jump_start = None
        self.opponent_jump_seen_count = 0
        self.tactics = TacticalMemory()

    def act(self, observation, command_index, rng):
        """Stable fallback for callers that do not provide resolver context."""
        if self.pending_response is not None:
            result = _legal_action(
                observation, command_index,
                [self.pending_response, "guard:crouching"],
            )
            if result == command_index.get(self.pending_response):
                self.pending_response = None
            return result
        return _legal_action(observation, command_index, ["guard:crouching"])

    def act_with_context(self, observation, command_index, rng, env, actor: int = 1):
        state = env.core.combat_state
        own = state.players[actor]
        opponent = state.players[1 - actor]

        deferred_punish = claim_deferred_punish(env, self.tactics, actor)
        if deferred_punish is not None:
            self.pending_response = f"immediate:{deferred_punish}"
            self.current_family = "punish"
            self.recent_events.append(
                f"guaranteed_punish_deferred:{opponent.action_id}"
            )

        if self.tactics.pending_challenge is not None:
            if self.tactics.challenge_delay > 0:
                self.tactics.challenge_delay -= 1
                return _legal_action(
                    observation, command_index, ["guard:crouching"]
                )
            self.pending_response = f"immediate:{self.tactics.pending_challenge}"
            self.tactics.pending_challenge = None

        if self.pending_response is not None:
            command = self.pending_response
            if own.action_id is not None and command == "immediate:drive_impact":
                command = "cancel:drive_impact"
            result = _legal_action(
                observation, command_index,
                [command, self.pending_response, "guard:crouching"],
            )
            if result in {
                command_index.get(command),
                command_index.get(self.pending_response),
            }:
                self.pending_response = None
            return result

        if opponent.action_id == "drive_impact":
            start_frame = state.frame - opponent.action_frame
            if self.seen_di_start_frame != start_frame:
                self.seen_di_start_frame = start_frame
                self.opponent_di_seen_count += 1
                self._choose_di_response(opponent.action_frame, state.distance, rng)
            if self.pending_response is not None:
                return self.act_with_context(
                    observation, command_index, rng, env, actor
                )

        opponent_movement = env._movements[1 - actor]
        if opponent_movement is not None and "jump" in opponent_movement.kind:
            if opponent_movement.frame <= 2:
                return _legal_action(
                    observation, command_index, ["guard:crouching"]
                )
            start_frame = state.frame - opponent_movement.frame
            if self.seen_jump_start != start_frame:
                self.seen_jump_start = start_frame
                self.opponent_jump_seen_count += 1
                response = choose_anti_air(
                    env,
                    actor=actor,
                    jumper=1 - actor,
                    movement=opponent_movement,
                    jump_count=self.opponent_jump_seen_count,
                    rng=rng,
                )
                if response is not None:
                    self.pending_response = f"immediate:{response}"
                    self.current_family = "anti_air"
                    self.recent_events.append("anti_air_claim")
                    return self.act(observation, command_index, rng)
            return _legal_action(
                observation, command_index, ["guard:standing"]
            )

        next_frame = state.frame + 1
        if (
            not self.wakeup_decided
            and next_frame >= env.initial_advantage
            and state.distance <= 72.0
        ):
            self.wakeup_decided = True
            self.pending_response = self._choose_wakeup_response(rng)
            return self.act(observation, command_index, rng)

        throw_range = env._range("throw_forward", "standing")
        if state.distance > throw_range:
            if self.approach_family is None:
                self.approach_family = (
                    "jump" if rng.random() < 0.35 * self.biases["jump"] else "walk_throw"
                )
            command = (
                "immediate:forward_jump"
                if self.approach_family == "jump"
                else "immediate:forward_walk"
            )
            return _legal_action(observation, command_index, [command, "guard:crouching"])
        if self.approach_family == "walk_throw":
            self.approach_family = None
            self.current_family = "tech"
            return _legal_action(
                observation, command_index,
                ["immediate:throw_forward", "guard:crouching"],
            )
        self.approach_family = None

        if self.last_attacker_command.endswith("backwalk"):
            self.current_family = "mash"
            return _legal_action(
                observation, command_index,
                ["immediate:2MK", "guard:crouching"],
            )

        if self.normal_to_special_seen_count >= 2:
            probability = 0.5 if self.normal_to_special_seen_count == 2 else 0.9
            probability *= self.biases["di"]
            probability *= 0.5 ** self.attacker_counter_di_count
            if rng.random() < min(probability, 0.95):
                self.current_family = "di"
                return _legal_action(
                    observation, command_index,
                    ["immediate:drive_impact", "guard:crouching"],
                )

        return _legal_action(observation, command_index, ["guard:crouching"])

    def _choose_wakeup_response(self, rng) -> str:
        if self.successful_family in {"block", "tech", "reversal"} and rng.random() < self.persistence:
            family = self.successful_family
        else:
            families = np.asarray(["block", "tech", "reversal"])
            weights = np.asarray([
                0.40 * self.biases["block"],
                0.35 * self.biases["tech"],
                0.25 * self.biases["reversal"],
            ], dtype=float)
            weights /= weights.sum()
            family = str(rng.choice(families, p=weights))
        self.current_family = family
        return {
            "block": "guard:crouching",
            "tech": "schedule:2:throw_forward",
            "reversal": "immediate:shoryuken_OD",
        }[family]

    def _choose_di_response(self, action_frame: int, distance: float, rng) -> None:
        self.pending_response = "immediate:drive_impact"
        self.current_family = "di"
        self.recent_events.append("di_response")

    def observe(self, infos: dict[str, Any], env, actor: int = 1) -> None:
        opponent = 1 - actor
        own_info = infos[f"player_{actor}"]
        opponent_info = infos[f"player_{opponent}"]
        self.last_attacker_command = str(opponent_info.get("submitted_command", "noop"))

        for event in own_info.get("events", []):
            event_type = event.get("type")
            event_actor = event.get("actor")
            target = event.get("target")
            data = event.get("data", {})
            action = str(data.get("action", ""))
            if event_type == "Block" and event_actor == opponent and target == actor:
                if action in {"5LP", "2LP"}:
                    self.blocked_light_count += 1
                    self.recent_events.append(f"light_pressure_{self.blocked_light_count}")
                else:
                    self.blocked_light_count = 0

                if action.startswith("jinrai_") and self.last_blocked_action in {
                    "5MP", "5HP", "2MP", "2MK",
                }:
                    self.normal_to_special_seen_count += 1
                    self.recent_events.append("normal_to_special")
                self.last_blocked_action = action

                punish, assessment = find_guaranteed_punish(
                    env,
                    action,
                    int(data.get("active_index", 1)),
                    env.core.combat_state.distance,
                    env.core.combat_state.players[opponent].posture,
                )
                self.tactics.last_block = {
                    "action": action,
                    "frame": env.core.combat_state.frame,
                    "stun": int(data.get("stun", 0)),
                    "active_index": int(data.get("active_index", 1)),
                    "contact_action_frame": int(
                        env.core.combat_state.players[opponent].action_frame
                    ),
                    "assessment": assessment.state.value,
                }
                if punish is not None:
                    self.pending_response = f"immediate:{punish}"
                    self.current_family = "punish"
                    self.recent_events.append(f"guaranteed_punish:{action}")
                elif action == "jHP":
                    self.jhp_block_count += 1
                    self.pending_response = "guard:crouching"
                    self.current_family = "post_jhp_block"
                    self.recent_events.append(
                        f"post_jhp_block_response_{self.jhp_block_count}"
                    )

            if (
                event_type in {"ActionStarted", "ReversalStarted"}
                and event_actor == opponent
                and self.tactics.last_block is not None
            ):
                previous = self.tactics.last_block
                try:
                    startup = int(env.core.data.move(action)["startup"])
                except (KeyError, TypeError, ValueError):
                    startup = 10_000
                gap = (
                    env.core.combat_state.frame + startup - 1
                    - (int(previous["frame"]) + int(previous["stun"]))
                )
                key = (str(previous["action"]), action)
                repetition = self.tactics.continuation_counts.get(key, 0) + 1
                self.tactics.continuation_counts[key] = repetition
                challenge = choose_challenge(
                    env,
                    gap=gap - 2,
                    distance=env.core.combat_state.distance,
                    opponent_posture=env.core.combat_state.players[opponent].posture,
                    repetition=repetition,
                    rng=env._rng,
                )
                if challenge is not None:
                    self.tactics.pending_challenge = challenge
                    self.tactics.challenge_delay = 2
                    self.tactics.challenge_attempt = key
                    self.current_family = "challenge"
                    self.recent_events.append(f"challenge:{gap}:{action}")
                self.tactics.last_block = None

            if self.tactics.challenge_attempt is not None and event_type == "Hit":
                key = self.tactics.challenge_attempt
                if event_actor == actor and target == opponent:
                    self.tactics.continuation_counts[key] = min(
                        6, self.tactics.continuation_counts.get(key, 1) + 1
                    )
                    self.recent_events.append("challenge_success")
                    self.tactics.challenge_attempt = None
                elif event_actor == opponent and target == actor:
                    self.tactics.continuation_counts[key] = max(
                        1, self.tactics.continuation_counts.get(key, 1) - 1
                    )
                    self.recent_events.append("challenge_counter_hit")
                    self.tactics.challenge_attempt = None

            if event_type == "Hit" and event_actor == actor and target == opponent:
                if self.current_family in {
                    "mash", "challenge", "punish", "anti_air", "post_jhp_block",
                }:
                    self.pending_response = "immediate:forward_jump"
                    self.current_family = "escape_conversion"
                    self.recent_events.append("interrupt_escape_conversion")

            if event_type == "CounterDI" and event_actor == opponent:
                self.attacker_counter_di_count += 1
                self.biases["di"] = max(0.25, self.biases["di"] * 0.5)
                self.recent_events.append("attacker_counter_di")
            if event_type in {"Hit", "Throw"} and event_actor == opponent and target == actor:
                self.recent_events.append("defense_failed")
                if self.current_family in self.biases:
                    self.biases[self.current_family] = max(
                        0.5, self.biases[self.current_family] * 0.85
                    )
                if event_type == "Hit" and action == "jHP":
                    self.jhp_hit_count += 1
                    self.pending_response = "guard:crouching"
                    self.current_family = "post_jhp_hit"
                    self.recent_events.append(f"post_jhp_response_{self.jhp_hit_count}")
            if event_type == "ThrowWhiff" and event_actor == opponent:
                reaches = (
                    env.core.combat_state.distance
                    <= env._range("5LP", env.core.combat_state.players[opponent].posture)
                )
                self.pending_response = (
                    "immediate:5LP" if reaches else "immediate:2MK"
                )
                self.current_family = "mash"
                self.recent_events.append("throw_whiff_punish")

        reason = own_info.get("terminal_reason")
        if reason in {
            "throw_tech", "side_switch", "pressure_escape", "counter_di_escape",
            "defender_disengage", "corner_escape",
        }:
            self.successful_family = self.current_family or "block"
            if self.successful_family in self.biases:
                self.biases[self.successful_family] = min(
                    1.5, self.biases[self.successful_family] * 1.15
                )
            self.recent_events.append(f"success:{self.successful_family}")


class MixedBot(HeuristicDefenderBot):
    """Compatibility name for the stateful heuristic training opponent."""


MashBot = AlwaysMashBot
ThrowBot = AlwaysThrowBot
ReversalBot = AlwaysODDPBot

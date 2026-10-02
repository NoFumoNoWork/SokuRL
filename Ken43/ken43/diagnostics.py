from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict
from typing import Any


COMBAT_EVENT_TYPES = {
    "Hit", "Block", "Whiff", "Throw", "ThrowWhiff", "ThrowClash",
    "ArmorAbsorb", "ArmorBreak", "CounterDI",
}
CONTACT_TYPES = COMBAT_EVENT_TYPES - {"Whiff", "ThrowWhiff"}


class EpisodeDiagnostics:
    """Structured episode diagnostics with a bounded per-frame trace."""

    def __init__(self, trace_frames: int = 120):
        self.trace_frames = int(trace_frames)
        self.reset(43, 70.0)

    def reset(self, initial_advantage: int, initial_distance: float) -> None:
        self.initial_advantage = int(initial_advantage)
        self.initial_distance = float(initial_distance)
        self.trace: deque[dict[str, Any]] = deque(maxlen=self.trace_frames)
        self.timeline: list[dict[str, Any]] = []
        self.action_frequency = [Counter(), Counter()]
        self.contact_counts: Counter[str] = Counter()
        self.schedule_delay_histogram: Counter[str] = Counter()
        self.jinrai: Counter[str] = Counter()
        self.od_dp: Counter[str] = Counter()
        self.di: Counter[str] = Counter()
        self.frame_kill: Counter[str] = Counter()
        self.drive_sources = [Counter(), Counter()]
        self.reward_totals = [Counter(), Counter()]
        self._open_jinrai: list[str | None] = [None, None]
        self._open_od_dp: list[dict[str, Any] | None] = [None, None]
        self._od_dp_drive_deltas: list[float] = []
        self._regen_active = [False, False]
        self.last_contact_frame = 0
        self.last_offensive_action_frame = 0
        self.last_position_change_frame = 0

    def timing_event(self, kind: str, frame: int, actor: int, **data: Any) -> None:
        item = {"type": kind, "frame": int(frame), "actor": int(actor), **data}
        self.timeline.append(item)
        if kind == "schedule_claim":
            key = f"target={data['target_delay']},noise={data['noise']},actual={data['actual_delay']}"
            self.schedule_delay_histogram[key] += 1
        if kind == "jinrai_branch_claim":
            key = f"claimed:{data.get('branch', 'unknown')}:delay={data.get('claimed_delay', 0)}"
            self.jinrai[key] += 1
        elif kind == "jinrai_branch_trigger":
            key = f"actual:{data.get('branch', 'unknown')}:delay={data.get('actual_delay', 0)}"
            self.jinrai[key] += 1

    def record_step(
        self,
        *,
        state,
        frame_data,
        labels: list[str],
        events,
        before_drive: list[float],
        resolved_drive: list[float],
        final_drive: list[float],
        before_distance: float,
        before_corner_offset: float,
        before_player_state: list[dict[str, Any]],
        corner_offset: float,
        disengaged: bool,
        disengage_cause: str | None,
        movements,
        pending,
        pending_branch,
        reward_drive: list[float],
        reward_bonus: list[float],
        rewards: list[float],
    ) -> list[dict[str, Any]]:
        frame = int(state.frame)
        step_events: list[dict[str, Any]] = []
        for actor, label in enumerate(labels):
            self.action_frequency[actor][label] += 1

        for event in events:
            payload = asdict(event)
            event_type = event.type
            action = str(event.data.get("action", ""))
            if event_type in COMBAT_EVENT_TYPES:
                if event_type in CONTACT_TYPES:
                    self.last_contact_frame = frame
                key = event_type if not action else f"{event_type}:{action}"
                self.contact_counts[key] += 1
                item = {"type": "contact", "frame": frame, **payload}
                self.timeline.append(item)
                step_events.append(item)
            if event_type in {"ActionStarted", "ReversalStarted"} and event.actor is not None:
                move = frame_data.move(action) if action in frame_data.moves else {}
                if move.get("hit_level") not in (None, "non_attack"):
                    self.last_offensive_action_frame = frame
                if action.startswith("jinrai_"):
                    strength = action.rsplit("_", 1)[-1]
                    self._open_jinrai[event.actor] = strength
                    self.jinrai[f"root:{strength}"] += 1
                if action == "shoryuken_OD":
                    self.od_dp["attempt"] += 1
                    self._open_od_dp[event.actor] = {
                        "frame": frame,
                        "drive_before": float(before_drive[event.actor]),
                    }
                if action == "drive_impact":
                    self.di["attempt"] += 1
                if action in {"5MK", "5HK", "2HK"}:
                    self.frame_kill[f"attempt:{action}"] += 1
            if event_type == "BranchStarted" and event.actor is not None:
                parent = self._open_jinrai[event.actor]
                child = str(event.data.get("action", event.data.get("child", "unknown")))
                if parent is not None:
                    self.jinrai[f"branch:{parent}:{child}"] += 1
                    self._open_jinrai[event.actor] = None
            if event_type == "ActionEnded" and event.actor is not None:
                ended = str(event.data.get("action", ""))
                if ended.startswith("jinrai_") and self._open_jinrai[event.actor] is not None:
                    self.jinrai[f"no_branch:{self._open_jinrai[event.actor]}"] += 1
                    self._open_jinrai[event.actor] = None
            self._record_move_outcome(event_type, action, event.data, event.actor)
            if (
                action == "shoryuken_OD"
                and event_type in {"Hit", "Block", "Whiff"}
                and event.actor is not None
                and self._open_od_dp[event.actor] is not None
            ):
                opened = self._open_od_dp[event.actor]
                self._od_dp_drive_deltas.append(
                    float(resolved_drive[event.actor]) - float(opened["drive_before"])
                )
                self._open_od_dp[event.actor] = None
            if event_type == "SpecialStun" and event.data.get("kind") == "wall_splat":
                self.di["wall_splat"] += 1
            if (
                event_type == "ArmorAbsorb"
                and event.actor is not None
                and state.players[event.actor].action_id == "drive_impact"
            ):
                self.di["armor_absorb"] += 1
            self._record_drive_source(event)

        for actor in range(2):
            regen = max(0.0, final_drive[actor] - resolved_drive[actor])
            if regen:
                self.drive_sources[actor]["regen"] += regen
                if not self._regen_active[actor]:
                    item = {
                        "type": "drive_regen_started",
                        "frame": frame,
                        "actor": actor,
                        "frames_since_last_contact": frame - self.last_contact_frame,
                    }
                    self.timeline.append(item)
                    step_events.append(item)
                self._regen_active[actor] = True
            else:
                self._regen_active[actor] = False
            self.reward_totals[actor]["drive"] += float(reward_drive[actor])
            self.reward_totals[actor]["event_bonus"] += float(reward_bonus[actor])
            self.reward_totals[actor]["total"] += float(rewards[actor])

        if state.distance != before_distance or corner_offset != before_corner_offset:
            self.last_position_change_frame = frame

        trace_item = {
            "frame": frame,
            "distance": float(state.distance),
            "distance_delta": float(state.distance - before_distance),
            "initial_advantage": self.initial_advantage,
            "corner": {
                "defender_at_corner": corner_offset <= 0.0,
                "defender_corner_offset": float(corner_offset),
            },
            "disengaged": bool(disengaged),
            "disengage_cause": disengage_cause,
            "submitted": list(labels),
            "players_before": deepcopy(before_player_state),
            "players": [
                self._player_snapshot(
                    actor, state.players[actor], frame_data,
                    movements[actor], pending[actor], pending_branch[actor],
                )
                for actor in range(2)
            ],
            "events": [asdict(event) for event in events],
            "drive": {
                "before": list(map(float, before_drive)),
                "after_resolver": list(map(float, resolved_drive)),
                "after_regen": list(map(float, final_drive)),
            },
            "reward": [
                {
                    "drive": float(reward_drive[actor]),
                    "event_bonus": float(reward_bonus[actor]),
                    "total": float(rewards[actor]),
                }
                for actor in range(2)
            ],
            "frames_since": {
                "last_contact": frame - self.last_contact_frame,
                "last_offensive_action": frame - self.last_offensive_action_frame,
                "last_position_change": frame - self.last_position_change_frame,
            },
        }
        self.trace.append(trace_item)
        return step_events

    def summary(
        self,
        terminal_reason: str,
        state,
        movements,
        pending,
        pending_branch,
        disengaged: bool,
        disengage_cause: str | None,
    ) -> dict[str, Any]:
        committed = {
            "active_action": any(player.action_id is not None for player in state.players),
            "movement": any(item is not None for item in movements),
            "scheduled_action": any(item is not None for item in pending),
            "branch_pending": any(item is not None for item in pending_branch),
        }
        committed["any"] = any(committed.values())
        od_dp = dict(self.od_dp)
        od_dp["unresolved_or_interrupted"] = max(
            0,
            int(od_dp.get("attempt", 0))
            - int(od_dp.get("hit", 0))
            - int(od_dp.get("block", 0))
            - int(od_dp.get("whiff", 0)),
        )
        od_dp["mean_net_drive_completed"] = (
            sum(self._od_dp_drive_deltas) / len(self._od_dp_drive_deltas)
            if self._od_dp_drive_deltas else 0.0
        )
        od_dp["completed_drive_samples"] = len(self._od_dp_drive_deltas)
        od_dp["net_drive_total_completed"] = sum(self._od_dp_drive_deltas)
        di = dict(self.di)
        di["unresolved_or_interrupted"] = max(
            0,
            int(di.get("attempt", 0))
            - int(di.get("hit", 0))
            - int(di.get("block", 0))
            - int(di.get("whiff", 0))
            - int(di.get("counter_di", 0)),
        )
        timeout_analysis = self._timeout_analysis() if terminal_reason == "max_frames" else None
        return {
            "terminal_reason": terminal_reason,
            "frames": int(state.frame),
            "initial_advantage": self.initial_advantage,
            "initial_distance": self.initial_distance,
            "final_distance": float(state.distance),
            "final_drive": [float(player.drive) for player in state.players],
            "disengaged": bool(disengaged),
            "disengage_cause": disengage_cause,
            "frames_since_last_contact": int(state.frame - self.last_contact_frame),
            "frames_since_last_offensive_action": int(state.frame - self.last_offensive_action_frame),
            "frames_since_last_position_change": int(state.frame - self.last_position_change_frame),
            "activity_at_end": committed,
            "timeout_analysis": timeout_analysis,
            "action_frequency": [dict(item) for item in self.action_frequency],
            "contact_events": dict(self.contact_counts),
            "schedule_delay_histogram": dict(self.schedule_delay_histogram),
            "jinrai": dict(self.jinrai),
            "od_dp": od_dp,
            "di": di,
            "frame_kill": dict(self.frame_kill),
            "drive_change_sources": [dict(item) for item in self.drive_sources],
            "reward_totals": [dict(item) for item in self.reward_totals],
            "timing_and_contact_timeline": deepcopy(self.timeline),
            "last_trace": list(self.trace),
        }

    def _timeout_analysis(self) -> dict[str, Any]:
        trace = list(self.trace)
        if not trace:
            return {"classification": "hard_stall"}
        first_drive = trace[0]["drive"]["before"]
        final_drive = trace[-1]["drive"]["after_regen"]
        net = [float(final_drive[i] - first_drive[i]) for i in range(2)]
        gross = sum(
            abs(float(item["drive"]["after_regen"][i] - item["drive"]["before"][i]))
            for item in trace for i in range(2)
        )
        regen = sum(
            max(0.0, float(item["drive"]["after_regen"][i] - item["drive"]["after_resolver"][i]))
            for item in trace for i in range(2)
        )
        contacts = sum(
            event["type"] in CONTACT_TYPES
            for item in trace for event in item["events"]
        )
        offensive_starts = sum(
            event["type"] in {"ActionStarted", "ReversalStarted"}
            and event.get("data", {}).get("action") not in {None, ""}
            for item in trace for event in item["events"]
        )
        position_changes = sum(abs(float(item["distance_delta"])) > 1e-9 for item in trace)
        labels = [label for item in trace for label in item["submitted"]]
        conservation = sum(
            label in {
                "immediate:noop", "immediate:guard:standing", "immediate:guard:crouching",
                "immediate:5LP", "immediate:2LP", "immediate:5LK", "immediate:2LK",
            }
            for label in labels
        ) / max(1, len(labels))
        guard = sum("guard:" in label for label in labels) / max(1, len(labels))
        hard_stall = contacts == 0 and gross < 1e-9 and position_changes == 0 and offensive_starts == 0
        soft_loop = (
            not hard_stall
            and regen > 0.0
            and contacts > 0
            and sum(abs(value) for value in net) < 0.25
            and conservation >= 0.80
        )
        classification = "hard_stall" if hard_stall else "soft_loop" if soft_loop else "live_combat"
        return {
            "classification": classification,
            "window_frames": len(trace),
            "net_drive_change": net,
            "gross_drive_change": float(gross),
            "regen_total": float(regen),
            "contact_events": int(contacts),
            "offensive_action_starts": int(offensive_starts),
            "position_change_frames": int(position_changes),
            "conservation_action_ratio": float(conservation),
            "fallback_guard_ratio": float(guard),
        }

    def _record_move_outcome(
        self, event_type: str, action: str, data: dict[str, Any], actor: int | None
    ) -> None:
        outcome = event_type.lower()
        if action == "shoryuken_OD" and event_type in {"Hit", "Block", "Whiff"}:
            self.od_dp[outcome] += 1
            if event_type == "Hit" and actor == 1:
                self.od_dp["pressure_escape"] += 1
        if action == "drive_impact":
            if event_type in {"Hit", "Block", "Whiff", "ArmorAbsorb", "ArmorBreak", "CounterDI"}:
                key = "counter_di" if event_type == "CounterDI" else outcome
                self.di[key] += 1
        if action in {"5MK", "5HK", "2HK"} and event_type == "Whiff":
            self.frame_kill[f"whiff:{action}"] += 1

    def _record_drive_source(self, event) -> None:
        actor, target, data = event.actor, event.target, event.data
        if event.type in {"ActionStarted", "ReversalStarted"} and actor is not None:
            cost = float(data.get("drive_cost", 0.0) or 0.0)
            if cost:
                self.drive_sources[actor][f"expenditure:{data.get('action', 'unknown')}"] -= cost
        if event.type in {"Hit", "Throw"}:
            gain = float(data.get("drive_gain_attacker", 0.0) or 0.0)
            loss = float(data.get("drive_damage_defender", 0.0) or 0.0)
            if actor is not None and gain:
                self.drive_sources[actor]["hit_or_throw_gain"] += gain
            if target is not None and loss:
                self.drive_sources[target]["hit_drive_damage"] -= loss
        if event.type == "Block" and target is not None:
            loss = float(data.get("drive_damage_defender", 0.0) or 0.0)
            if loss:
                self.drive_sources[target]["block_chip"] -= loss
        if event.type == "ThrowClash":
            gain = float(data.get("drive_gain_each", 0.0) or 0.0)
            for index in range(2):
                self.drive_sources[index]["throw_tech_gain"] += gain

    @staticmethod
    def _player_snapshot(actor, fighter, frame_data, movement, pending, pending_branch):
        action = fighter.action_id
        phase = "idle"
        if movement is not None:
            phase = "movement"
        elif action is not None:
            active = frame_data.move(action).get("active", [])
            active_frames = [frame for start, end in active for frame in range(int(start), int(end) + 1)]
            if active_frames and fighter.action_frame < min(active_frames):
                phase = "startup"
            elif fighter.action_frame in active_frames:
                phase = "active"
            else:
                phase = "recovery"
        return {
            "actor": actor,
            "facing": "right" if actor == 0 else "left",
            "drive": float(fighter.drive),
            "posture": fighter.posture,
            "airborne": bool(fighter.airborne),
            "blocking": bool(fighter.blocking),
            "action": action,
            "action_frame": int(fighter.action_frame),
            "phase": phase,
            "movement": asdict(movement) if movement is not None else None,
            "schedule": asdict(pending) if pending is not None else None,
            "jinrai_branch_claim": asdict(pending_branch) if pending_branch is not None else None,
        }

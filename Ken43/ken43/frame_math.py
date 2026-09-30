from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any, Literal

from .data import FrameData, load_frame_data

Outcome = Literal["hit", "block"]


@dataclass(frozen=True)
class Contact:
    move_id: str
    outcome: Outcome
    contact_active_index: int
    advantage: int


@dataclass(frozen=True)
class JumpContact:
    contact: bool
    advantage: int | None
    remaining_usable_active: int | None
    reason: str


@dataclass(frozen=True)
class ThrowAttempt:
    captures: bool
    first_active_relative_to_wakeup: int
    reason: str


@dataclass(frozen=True)
class RaceResult:
    attacker_active_frame: int
    defender_active_frame: int
    result: str


@dataclass(frozen=True)
class CancelTiming:
    gap: int
    overlap: int
    true_blockstring: bool


class FrameCalculator:
    """Frame-rule calculator for the Ken oki microgame.

    All public methods derive timing from the JSON frame data and the shared
    rule conventions in that document.
    """

    def __init__(self, frame_data: FrameData | None = None):
        self.data = frame_data or load_frame_data()

    def active_frames(self, move_id: str) -> list[int]:
        move = self.data.move(move_id)
        frames: list[int] = []
        active_windows = move.get(
            "active",
            move.get("active_relative_to_branch", move.get("active_relative_to_input", [])),
        )
        for start, end in active_windows:
            frames.extend(range(int(start), int(end) + 1))
        return frames

    def active_index_at_action_frame(self, move_id: str, action_frame: int) -> int | None:
        for index, frame in enumerate(self.active_frames(move_id), start=1):
            if frame == action_frame:
                return index
        return None

    def remaining_active_after(self, move_id: str, contact_active_index: int) -> int:
        total_active = len(self.active_frames(move_id))
        if not 1 <= contact_active_index <= total_active:
            raise ValueError(
                f"{move_id} active index {contact_active_index} is outside 1..{total_active}"
            )
        return total_active - contact_active_index

    def action_total(self, move_id: str) -> int:
        move = self.data.move(move_id)
        if "total" in move:
            return int(move["total"])
        active = self.active_frames(move_id)
        if active:
            if "recovery" in move:
                return max(active) + int(move["recovery"])
            if "landing_recovery" in move:
                return max(active) + int(move["landing_recovery"])
        if "startup" in move and "recovery" in move:
            return int(move["startup"]) + int(move["recovery"])
        raise ValueError(f"{move_id} has no total-frame model")

    def stun(self, move_id: str, outcome: Outcome) -> int:
        value = self.data.move(move_id).get("stun", {}).get(outcome)
        if value is None:
            raise ValueError(f"{move_id} has no {outcome} stun in V0")
        return int(value)

    def raw_advantage(
        self, move_id: str, outcome: Outcome, contact_active_index: int = 1
    ) -> int:
        move = self.data.move(move_id)
        remaining = self.remaining_active_after(move_id, contact_active_index)
        return self.stun(move_id, outcome) - (remaining + int(move["recovery"]) + 1)

    def contact(
        self, move_id: str, outcome: Outcome, contact_active_index: int = 1
    ) -> Contact:
        return Contact(
            move_id=move_id,
            outcome=outcome,
            contact_active_index=contact_active_index,
            advantage=self.raw_advantage(move_id, outcome, contact_active_index),
        )

    def cancel_gap(
        self,
        parent_id: str,
        child_id: str,
        outcome: Outcome,
        delay: int = 0,
    ) -> int:
        if delay < 0:
            raise ValueError("cancel delay must be non-negative")
        return int(self.data.move(child_id)["startup"]) + delay - self.stun(parent_id, outcome)

    def cancel_timing(
        self,
        parent_id: str,
        child_id: str,
        outcome: Outcome,
        delay: int = 0,
    ) -> CancelTiming:
        gap = self.cancel_gap(parent_id, child_id, outcome, delay)
        return CancelTiming(
            gap=gap,
            overlap=max(0, -gap),
            true_blockstring=outcome == "block" and gap < 0,
        )

    def cancel_gap_after_contact(
        self,
        parent_id: str,
        child_id: str,
        outcome: Outcome,
        contact_active_index: int,
        delay: int = 0,
    ) -> int:
        self.remaining_active_after(parent_id, contact_active_index)
        return self.cancel_gap(parent_id, child_id, outcome, delay)

    def is_cancel_combo(
        self,
        parent_id: str,
        child_id: str,
        outcome: Outcome,
        delay: int = 0,
    ) -> bool:
        return self.cancel_gap(parent_id, child_id, outcome, delay) < 0

    def quick_dash_fastest_emergency_stop_total(self) -> int:
        followup = self.data.move("quick_dash_emergency_stop")
        if "actionable_on_quick_dash_frame" in followup:
            return int(followup["actionable_on_quick_dash_frame"])
        branch = self.data.move("quick_dash")["branches"]["emergency_stop"]
        return int(branch["input_frame"]) + int(followup["startup_from_branch_input"])

    def quick_dash_followup_first_active(self, followup_id: str) -> int:
        followup = self.data.move(followup_id)
        if "first_active_from_quick_dash_start" in followup:
            return int(followup["first_active_from_quick_dash_start"])
        branch_info = followup.get("parent_branch", {})
        if "first_active_from_quick_dash_start" in branch_info:
            return int(branch_info["first_active_from_quick_dash_start"])
        parent = self.data.move("quick_dash")
        for branch in parent.get("branches", {}).values():
            if branch.get("result_action") == followup_id:
                startup = followup.get("startup", followup.get("startup_from_branch"))
                return int(branch["input_frame"]) + int(startup)
        raise ValueError(f"quick_dash has no branch to {followup_id}")

    def plus43_5lp_whiff_qd_thunder_contact(self) -> Contact:
        first_active_absolute = self.action_total("5LP") + self.quick_dash_followup_first_active(
            "quick_dash_thunder_kick"
        )
        wakeup_frame = 43
        contact_active_index = wakeup_frame - first_active_absolute + 1
        return self.contact("quick_dash_thunder_kick", "hit", contact_active_index)

    def safe_jump_jhp(self, input_jump_frame: int, knockdown_advantage: int = 42) -> JumpContact:
        jhp = self.data.move("jHP")
        calibration = self.data.jump["jHP_setup_calibration"][f"plus{knockdown_advantage}_safe_jump"]
        reference_input = self._last_usable_jhp_reference_input(calibration)
        active_start, active_end = jhp["active_relative_to_input"][0]
        active_duration = int(active_end) - int(active_start) + 1
        earliest_touching_input = reference_input - active_duration + 1

        if input_jump_frame > reference_input:
            return JumpContact(False, None, None, "jHP has no usable active frame before landing")
        if input_jump_frame < earliest_touching_input:
            return JumpContact(
                False,
                knockdown_advantage - int(self.data.jump["forward_jump_total_to_actionable"]),
                None,
                "jHP active window ended before the defender became vulnerable",
            )

        remaining = reference_input - input_jump_frame
        advantage = int(jhp["stun"]["block"]) - (remaining + int(jhp["landing_recovery"]))
        return JumpContact(True, advantage, remaining, "jHP contacts during the usable safe-jump window")

    def _last_usable_jhp_reference_input(self, calibration: dict[str, Any]) -> int:
        for key, value in calibration.items():
            if isinstance(value, dict) and value.get("contact") == "last_usable_active_frame":
                match = re.search(r"input_at_jump_frame_(\d+)", key)
                if match:
                    return int(match.group(1))
        raise ValueError("jHP calibration lacks a last-usable-active reference")

    def jinrai_followup_first_active_root_frame(self, root_id: str, followup_id: str) -> int:
        root = self.data.move(root_id)
        empirical = root.get("empirical_branch_timing", {})
        if followup_id == "kazekama" and "fastest_LK_followup_first_active_root_frame" in empirical:
            return int(empirical["fastest_LK_followup_first_active_root_frame"])
        for branch in root.get("branches", {}).values():
            if branch.get("child") == followup_id:
                earliest_input = int(branch["input_window"][0])
                # Root frame 1 has global offset 0, and the input frame itself
                # is child frame 1 rather than a transition-only frame.
                return earliest_input + int(self.data.move(followup_id)["startup"]) - 2
        raise ValueError(f"{root_id} has no branch to {followup_id}")

    def jinrai_followup_contact_on_wakeup(
        self,
        root_id: str,
        followup_id: str,
        knockdown_advantage: int,
        guard: Literal["standing", "crouching"],
    ) -> Contact:
        first_active = self.jinrai_followup_first_active_root_frame(root_id, followup_id)
        contact_active_index = knockdown_advantage - first_active + 1
        move = self.data.move(followup_id)
        hit_level = move["hit_level"]
        outcome: Outcome = "block"
        if hit_level == "low" and guard == "standing":
            outcome = "hit"
        elif hit_level == "overhead" and guard == "crouching":
            outcome = "hit"
        return self.contact(followup_id, outcome, contact_active_index)

    def jinrai_followup_vs_defender_normal(
        self,
        root_id: str,
        followup_id: str,
        defender_startup: int,
        root_contact_action_frame: int | None = None,
    ) -> RaceResult:
        root_contact = root_contact_action_frame or self.active_frames(root_id)[0]
        defender_first_actionable = root_contact + self.stun(root_id, "block")
        defender_active = defender_first_actionable + defender_startup - 1
        attacker_active = self.jinrai_followup_first_active_root_frame(root_id, followup_id)
        if attacker_active < defender_active:
            result = "ATTACKER_HITS_FIRST"
        elif attacker_active > defender_active:
            result = "DEFENDER_HITS_FIRST"
        else:
            result = "SAME_FRAME"
        return RaceResult(attacker_active, defender_active, result)

    def plus_throw_attempt(self, plus_frames: int, throw_id: str = "throw_forward") -> ThrowAttempt:
        startup = int(self.data.move(throw_id)["startup"])
        first_active_relative_to_wakeup = plus_frames - startup
        invul = int(self.data.system_rules["wakeup_throw_invulnerability"])
        if first_active_relative_to_wakeup <= 0:
            return ThrowAttempt(False, first_active_relative_to_wakeup, "throw becomes active before wakeup")
        if first_active_relative_to_wakeup <= invul:
            return ThrowAttempt(
                False,
                first_active_relative_to_wakeup,
                "first active capture collides with wakeup throw invulnerability and whiffs",
            )
        return ThrowAttempt(True, first_active_relative_to_wakeup, "throw captures")

    def delayed_execution(self, target_delay: int, rng: random.Random | None = None) -> int:
        if target_delay < 0:
            raise ValueError("target delay must be non-negative")
        if target_delay == 0:
            return 0
        chooser = rng or random
        return max(0, target_delay + chooser.choice([-2, -1, 0, 1, 2]))

    def delayed_execution_outcomes(self, target_delay: int) -> list[int]:
        if target_delay < 0:
            raise ValueError("target delay must be non-negative")
        if target_delay == 0:
            return [0, 0, 0, 0, 0]
        return [max(0, target_delay + error) for error in (-2, -1, 0, 1, 2)]

    def quick_dash_thunder_contact_for_plus(
        self,
        knockdown_advantage: int,
        delay: int = 0,
        prefix_move_ids: tuple[str, ...] = (),
    ) -> Contact | None:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        prefix_total = sum(self.action_total(move_id) for move_id in prefix_move_ids)
        first_active = (
            prefix_total
            + self.quick_dash_followup_first_active("quick_dash_thunder_kick")
            + delay
        )
        active_count = len(self.active_frames("quick_dash_thunder_kick"))
        last_active = first_active + active_count - 1
        if knockdown_advantage > last_active:
            return None
        contact_active_index = max(1, knockdown_advantage - first_active + 1)
        return self.contact("quick_dash_thunder_kick", "hit", contact_active_index)

    def quick_dash_thunder_success_rate(
        self,
        knockdown_advantage: int,
        target_delay: int,
        minimum_advantage: int = 4,
        prefix_move_ids: tuple[str, ...] = (),
    ) -> float:
        successes = 0
        for actual_delay in self.delayed_execution_outcomes(target_delay):
            contact = self.quick_dash_thunder_contact_for_plus(
                knockdown_advantage,
                actual_delay,
                prefix_move_ids,
            )
            if contact is not None and contact.advantage >= minimum_advantage:
                successes += 1
        return successes / 5

    def enabled_action_ids(self) -> list[str]:
        """Moves with enough V0 data to execute, including contextual children."""
        explicitly_masked = set(
            self.data.system_rules.get("masked_actions_v0", [])
        )
        enabled: list[str] = []
        for move_id, move in self.data.moves.items():
            if move_id in explicitly_masked:
                continue
            if str(move.get("range_policy", "")).startswith("masked_in_V0"):
                continue
            enabled.append(move_id)
        return enabled

    def legal_action_ids(self) -> list[str]:
        """Moves that may be submitted independently through ``action``."""
        contextual_kinds = {
            "jinrai_followup",
            "quick_dash_followup",
            "target_combo_followup",
        }
        return [
            move_id
            for move_id in self.enabled_action_ids()
            if self.data.move(move_id).get("kind") not in contextual_kinds
        ]

    def simplified_hit_count(self, move_id: str) -> int:
        counts = self.data.system_rules.get("simplified_multi_hit_counts", {})
        return int(counts.get(move_id, 1))

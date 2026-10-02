from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .data import FrameData, load_frame_data

Posture = Literal["standing", "crouching"]
RecoveryMode = Literal["hold2", "non_hold2", "full", "cancelled"]


@dataclass(frozen=True)
class SpacingResult:
    move_id: str
    initial_distance: float
    defender_posture: Posture
    effective_range: float | None
    range_check: bool | None
    contact_or_whiff: Literal["block", "whiff"]
    spacing_transition: str
    final_distance: float
    next_move_legal_or_whiff: str


class SpacingCalculator:
    """Data-driven range checks and post-block distance transitions."""

    def __init__(self, frame_data: FrameData | None = None):
        self.data = frame_data or load_frame_data()

    def effective_range(self, move_id: str, posture: Posture) -> float:
        move = self.data.move(move_id)
        reach = move.get("attack_reach")
        if not isinstance(reach, (int, float)):
            measurement = self.data.system_rules.get("range_measurements", {}).get(move_id, {})
            first_whiff = measurement.get("standing_first_whiff")
            if not isinstance(first_whiff, (int, float)):
                ranges = move.get("effective_range")
                first_whiff = ranges.get("standing") if isinstance(ranges, dict) else None
            if not isinstance(first_whiff, (int, float)):
                raise ValueError(f"{move_id} has no measured attack reach")
            reach = float(first_whiff) - self._hurtbox_half_width("standing")
        return float(reach) + self._hurtbox_half_width(posture)

    def attack_reach(self, move_id: str) -> float:
        return self.effective_range(move_id, "standing") - self._hurtbox_half_width("standing")

    def _hurtbox_half_width(self, posture: Posture) -> float:
        geometry = self.data.system_rules.get("hurtbox_geometry", {})
        value = geometry.get(f"{posture}_half_width")
        if not isinstance(value, (int, float)):
            raise ValueError(f"missing {posture} hurtbox half width")
        return float(value)

    def in_range(self, move_id: str, distance: float, posture: Posture) -> bool:
        return float(distance) <= self.effective_range(move_id, posture)

    def spacing_after_block(
        self,
        move_id: str,
        distance: float,
        recovery_mode: RecoveryMode = "hold2",
    ) -> tuple[float, str]:
        spacing = self.data.move(move_id).get("spacing_on_block")
        if not isinstance(spacing, dict):
            raise ValueError(f"{move_id} has no block-spacing model")
        delta = float(spacing["delta"])
        shifted = float(distance) + delta
        if recovery_mode in ("hold2", "cancelled") and "D_min_cancelled_recovery" in spacing:
            key = "D_min_cancelled_recovery"
            floor = float(spacing[key])
            return max(floor, shifted), f"max({floor:.2f}, d + {delta:.2f})"
        if "D_min_full" in spacing:
            floor = float(spacing["D_min_full"])
            return max(floor, shifted), f"max({floor:.2f}, d + {delta:.2f})"
        return shifted, f"d + {delta:.2f}"

    def resolve_block(
        self,
        move_id: str,
        distance: float,
        posture: Posture,
        recovery_mode: RecoveryMode = "hold2",
        next_move_id: str | None = None,
        next_posture: Posture | None = None,
    ) -> SpacingResult:
        try:
            effective_range = self.effective_range(move_id, posture)
            range_check: bool | None = float(distance) <= effective_range
            contact = range_check
        except ValueError:
            effective_range = None
            range_check = None
            contact = True
        if contact:
            final_distance, transition = self.spacing_after_block(
                move_id, distance, recovery_mode
            )
        else:
            final_distance, transition = float(distance), "none (whiff)"

        next_result = "not_checked"
        if next_move_id is not None:
            next_guard = next_posture or posture
            next_result = "legal" if self.in_range(next_move_id, final_distance, next_guard) else "whiff"

        return SpacingResult(
            move_id=move_id,
            initial_distance=float(distance),
            defender_posture=posture,
            effective_range=effective_range,
            range_check=range_check,
            contact_or_whiff="block" if contact else "whiff",
            spacing_transition=transition,
            final_distance=final_distance,
            next_move_legal_or_whiff=next_result,
        )

    def backwalk_distance(self, initial_distance: float, frames: int) -> float:
        if frames < 0:
            raise ValueError("backwalk frames must be non-negative")
        if frames == 0:
            return float(initial_distance)
        backwalk = self.data.distance_model["backwalk"]
        return (
            float(initial_distance)
            + float(backwalk["first_frame_delta"])
            + (frames - 1) * float(backwalk["subsequent_frame_delta"])
        )

    def spacing_after_whiff(
        self,
        move_id: str,
        distance: float,
        recovery_mode: RecoveryMode = "hold2",
    ) -> tuple[float, float]:
        values = self.data.system_rules.get("whiff_forward_displacement", {})
        displacement = values.get(move_id, 0.0)
        if isinstance(displacement, dict):
            mode = "non_hold2" if recovery_mode in ("non_hold2", "full") else "hold2"
            displacement = displacement[mode]
        amount = float(displacement)
        return max(0.0, float(distance) - amount), amount

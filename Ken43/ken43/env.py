from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .data import FrameData, load_frame_data
from .frame_math import FrameCalculator
from .spacing import Posture, RecoveryMode, SpacingCalculator, SpacingResult


@dataclass
class FrameState:
    frame: int = 0
    attacker_action: str | None = None
    attacker_action_frame: int = 0
    defender_action: str | None = None
    defender_action_frame: int = 0
    distance: float = 0.0
    attacker_hp: int = 10000
    defender_hp: int = 10000
    attacker_drive: int = 6
    defender_drive: int = 6
    attacker_blocking: bool = False
    defender_blocking: bool = False
    attacker_airborne: bool = False
    defender_airborne: bool = False
    attacker_invulnerable: bool = False
    defender_invulnerable: bool = False
    attacker_knockdown: bool = False
    defender_knockdown: bool = True
    attacker_stun_remaining: int = 0
    defender_stun_remaining: int = 0
    attacker_throw_invul_remaining: int = 0
    defender_throw_invul_remaining: int = 0

    def observation(self) -> dict[str, Any]:
        return asdict(self)


class KenOkiMicrogame:
    """Small frame-advanced Ken-vs-Ken oki environment.

    The class intentionally keeps the environment narrow: fixed Ken vs Ken,
    corner oki, no juggle, no hitstop, no supers, and unsupported actions are
    masked by `action_mask`.
    """

    fps = 60

    def __init__(self, frame_data: FrameData | None = None):
        self.data = frame_data or load_frame_data()
        self.calc = FrameCalculator(self.data)
        self.spacing = SpacingCalculator(self.data)
        self.state = FrameState()
        self.knockdown_advantage = 43
        self._legal_actions = set(self.calc.legal_action_ids())

    def reset(
        self,
        *,
        knockdown_advantage: int = 43,
        distance: float = 0.0,
        attacker_hp: int = 10000,
        defender_hp: int = 10000,
        attacker_drive: int = 6,
        defender_drive: int = 6,
    ) -> dict[str, Any]:
        self.knockdown_advantage = knockdown_advantage
        self.state = FrameState(
            distance=distance,
            attacker_hp=attacker_hp,
            defender_hp=defender_hp,
            attacker_drive=attacker_drive,
            defender_drive=defender_drive,
            defender_knockdown=True,
        )
        return self.state.observation()

    def action_mask(self) -> dict[str, bool]:
        return {move_id: move_id in self._legal_actions for move_id in self.data.moves}

    def resolve_blocked_move(
        self,
        move_id: str,
        defender_posture: Posture,
        recovery_mode: RecoveryMode = "full",
        next_move_id: str | None = None,
        next_posture: Posture | None = None,
    ) -> SpacingResult:
        result = self.spacing.resolve_block(
            move_id,
            self.state.distance,
            defender_posture,
            recovery_mode,
            next_move_id,
            next_posture,
        )
        self.state.distance = result.final_distance
        self.state.defender_blocking = result.contact_or_whiff == "block"
        return result

    def start_attacker_action(self, action_id: str) -> None:
        if not self.action_mask().get(action_id, False):
            raise ValueError(f"action is masked in V0: {action_id}")
        self.state.attacker_action = action_id
        self.state.attacker_action_frame = 0
        self.state.attacker_airborne = self.data.move(action_id)["kind"] == "air_normal"

    def start_defender_action(self, action_id: str) -> None:
        if not self.action_mask().get(action_id, False):
            raise ValueError(f"action is masked in V0: {action_id}")
        self.state.defender_action = action_id
        self.state.defender_action_frame = 0
        self.state.defender_airborne = self.data.move(action_id)["kind"] == "air_normal"

    def step(
        self,
        attacker_action: str | None = None,
        defender_action: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if attacker_action is not None and self.state.attacker_action is None:
            self.start_attacker_action(attacker_action)
        if defender_action is not None and self.state.defender_action is None:
            self.start_defender_action(defender_action)

        self.state.frame += 1
        if self.state.attacker_action is not None:
            self.state.attacker_action_frame += 1
        if self.state.defender_action is not None:
            self.state.defender_action_frame += 1

        if self.state.frame >= self.knockdown_advantage:
            if self.state.defender_knockdown:
                self.state.defender_throw_invul_remaining = int(
                    self.data.system_rules["wakeup_throw_invulnerability"]
                )
            self.state.defender_knockdown = False

        self.state.attacker_stun_remaining = max(0, self.state.attacker_stun_remaining - 1)
        self.state.defender_stun_remaining = max(0, self.state.defender_stun_remaining - 1)
        self.state.attacker_throw_invul_remaining = max(
            0, self.state.attacker_throw_invul_remaining - 1
        )
        self.state.defender_throw_invul_remaining = max(
            0, self.state.defender_throw_invul_remaining - 1
        )

        terminated = self.state.attacker_hp <= 0 or self.state.defender_hp <= 0
        return self.state.observation(), 0.0, terminated, {"fps": self.fps}

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .data import FrameData, load_frame_data
from .frame_math import FrameCalculator
from .resolver import CombatState, FighterState, FrameEvent, FrameInput, FrameResolver
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
    attacker_posture: Posture = "standing"
    defender_posture: Posture = "standing"
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
    fixed_wakeup_frame = 43

    def __init__(self, frame_data: FrameData | None = None):
        self.data = frame_data or load_frame_data()
        self.calc = FrameCalculator(self.data)
        self.spacing = SpacingCalculator(self.data)
        self.resolver = FrameResolver(self.data)
        self.state = FrameState()
        self.combat_state = CombatState()
        self.knockdown_advantage = self.fixed_wakeup_frame
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
        if knockdown_advantage != self.fixed_wakeup_frame:
            raise ValueError(
                f"Ken43 V0 uses a fixed +{self.fixed_wakeup_frame} wakeup timeline"
            )
        self.knockdown_advantage = self.fixed_wakeup_frame
        self.state = FrameState(
            distance=distance,
            attacker_hp=attacker_hp,
            defender_hp=defender_hp,
            attacker_drive=attacker_drive,
            defender_drive=defender_drive,
            defender_knockdown=True,
        )
        self.combat_state = CombatState(
            distance=distance,
            players=[
                FighterState(hp=attacker_hp, drive=attacker_drive),
                FighterState(
                    hp=defender_hp,
                    drive=defender_drive,
                    knockdown=True,
                ),
            ],
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
        self.combat_state.distance = result.final_distance
        self.state.defender_blocking = result.contact_or_whiff == "block"
        return result

    def resolve_frame(
        self,
        attacker_input: FrameInput | None = None,
        defender_input: FrameInput | None = None,
    ) -> tuple[CombatState, list[FrameEvent]]:
        if (
            self.combat_state.players[1].knockdown
            and self.combat_state.frame + 1 >= self.knockdown_advantage
        ):
            defender = self.combat_state.players[1]
            defender.knockdown = False
            defender.throw_invul_remaining = int(
                self.data.system_rules["wakeup_throw_invulnerability"]
            )
        next_state, events = self.resolver.resolve_frame(
            self.combat_state,
            (attacker_input or FrameInput(), defender_input or FrameInput()),
        )
        self.combat_state = next_state
        self._sync_legacy_state()
        return next_state, events

    def _sync_legacy_state(self) -> None:
        attacker, defender = self.combat_state.players
        self.state.frame = self.combat_state.frame
        self.state.distance = self.combat_state.distance
        self.state.attacker_action = attacker.action_id
        self.state.attacker_action_frame = attacker.action_frame
        self.state.defender_action = defender.action_id
        self.state.defender_action_frame = defender.action_frame
        self.state.attacker_hp = attacker.hp
        self.state.defender_hp = defender.hp
        self.state.attacker_drive = attacker.drive
        self.state.defender_drive = defender.drive
        self.state.attacker_blocking = attacker.blocking
        self.state.defender_blocking = defender.blocking
        self.state.attacker_posture = attacker.posture
        self.state.defender_posture = defender.posture
        self.state.attacker_airborne = attacker.airborne
        self.state.defender_airborne = defender.airborne
        self.state.attacker_invulnerable = bool(
            self.resolver.active_invulnerability_types(attacker)
        )
        self.state.defender_invulnerable = bool(
            self.resolver.active_invulnerability_types(defender)
        )
        self.state.attacker_knockdown = attacker.knockdown
        self.state.defender_knockdown = defender.knockdown
        self.state.attacker_stun_remaining = attacker.stun_remaining
        self.state.defender_stun_remaining = defender.stun_remaining
        self.state.attacker_throw_invul_remaining = attacker.throw_invul_remaining
        self.state.defender_throw_invul_remaining = defender.throw_invul_remaining

    def start_attacker_action(self, action_id: str) -> None:
        if not self.action_mask().get(action_id, False):
            raise ValueError(f"action is masked in V0: {action_id}")
        self.combat_state.players[0].queued_action = action_id
        self.state.attacker_action = action_id
        self.state.attacker_action_frame = 0
        self.state.attacker_airborne = self.data.move(action_id)["kind"] == "air_normal"

    def start_defender_action(self, action_id: str) -> None:
        if not self.action_mask().get(action_id, False):
            raise ValueError(f"action is masked in V0: {action_id}")
        self.combat_state.players[1].queued_action = action_id
        self.state.defender_action = action_id
        self.state.defender_action_frame = 0
        self.state.defender_airborne = self.data.move(action_id)["kind"] == "air_normal"

    def step(
        self,
        attacker_action: str | None = None,
        defender_action: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        before_attacker_hp = self.combat_state.players[0].hp
        before_defender_hp = self.combat_state.players[1].hp
        _, events = self.resolve_frame(
            FrameInput(action=attacker_action),
            FrameInput(action=defender_action),
        )
        terminated = self.state.attacker_hp <= 0 or self.state.defender_hp <= 0
        reward = (before_defender_hp - self.state.defender_hp) - (
            before_attacker_hp - self.state.attacker_hp
        )
        return self.state.observation(), float(reward), terminated, {
            "fps": self.fps,
            "events": [asdict(event) for event in events],
        }

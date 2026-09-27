"""Player state decoded from game memory."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerState:
    x: float
    y: float
    action_id: int
    hp: int
    spirit: float
    airborne: bool

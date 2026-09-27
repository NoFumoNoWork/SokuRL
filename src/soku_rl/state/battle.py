"""Battle-level state shared by both players."""

from dataclasses import dataclass

from .player import PlayerState


@dataclass(frozen=True, slots=True)
class BattleState:
    player1: PlayerState
    player2: PlayerState
    frame: int

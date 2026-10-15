"""Validate the two character and profile-deck selections before game launch."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PlayerSetup:
    character: int
    palette: int
    deck: int

    def __post_init__(self):
        for name, limit in (("character", 20), ("palette", 8), ("deck", 4)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < limit:
                raise ValueError(f"{name} must be an integer in [0, {limit})")


@dataclass(frozen=True)
class MatchConfig:
    player_0: PlayerSetup
    player_1: PlayerSetup

    def __post_init__(self):
        for name in ("player_0", "player_1"):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = PlayerSetup(**value)
                object.__setattr__(self, name, value)
            if not isinstance(value, PlayerSetup):
                raise TypeError(f"{name} must be a PlayerSetup")

    def environment(self):
        return {f"SOKURL_VS_P{seat}_{field.upper()}": str(value)
                for seat, player in enumerate((self.player_0, self.player_1), 1)
                for field, value in asdict(player).items()}


# Exact configuration of checkpoints written before explicit match selection.
LEGACY_MATCH = MatchConfig(PlayerSetup(1, 0, 0), PlayerSetup(0, 0, 0))

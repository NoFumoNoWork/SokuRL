"""Native match identity and score transport contract."""
from dataclasses import dataclass

@dataclass(frozen=True)
class MatchState:
    """Transport-independent match identity, simulation time, scores and health."""
    match: int
    round: int
    frame: int
    scores: tuple[int, int]
    hp: tuple[int, int]
    phase: str

    def __post_init__(self):
        if self.phase not in {"battle", "menu", "loading", "disconnected"}:
            raise ValueError("unsupported match phase")
        if any(type(value) is not int or value < 0 for value in (self.match, self.round, self.frame)):
            raise ValueError("match, round and frame must be nonnegative integers")
        if self.phase == "battle" and self.match == 0:
            raise ValueError("battle requires a positive match identifier")
        if (len(self.scores) != 2 or len(self.hp) != 2
                or any(type(value) is not int or value < 0 for value in self.scores)
                or any(type(value) is not int for value in self.hp)):
            raise ValueError("match state requires two integer scores and health values")

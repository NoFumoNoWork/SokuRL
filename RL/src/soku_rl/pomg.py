"""Two-player partial-observation game contracts, independent of the engine."""
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Generic, Mapping, Protocol, TypeVar


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class Outcome(str, Enum):
    ONGOING = "ongoing"
    P1_WIN = "p1_win"
    P2_WIN = "p2_win"
    DRAW = "double_ko"
    TRUNCATED = "time_limit"


@dataclass(frozen=True, slots=True)
class TimeStep(Generic[ObservationT]):
    frame: int
    observations: tuple[ObservationT, ObservationT]
    rewards: tuple[float, float]
    outcome: Outcome
    diagnostics: Mapping[str, object]

    def __post_init__(self):
        if self.frame < 0 or len(self.observations) != 2 or len(self.rewards) != 2:
            raise ValueError("a time step needs a nonnegative frame and two players")
        if not all(isfinite(r) for r in self.rewards):
            raise ValueError("rewards must be finite")
        if not isinstance(self.outcome, Outcome):
            raise ValueError("outcome must be an Outcome")

    @property
    def ended(self):
        return self.outcome != Outcome.ONGOING

    @property
    def terminated(self):
        return self.outcome in {Outcome.P1_WIN, Outcome.P2_WIN, Outcome.DRAW}

    @property
    def truncated(self):
        return self.outcome == Outcome.TRUNCATED


class GameBatch(Protocol[ObservationT, ActionT]):
    """Slots evolve independently. Step accepts exactly the still-active slots.

    reset creates new episodes; it does not promise engine save-state support.
    Implementations must clean up partial resets when close is called.
    """

    def reset(self, seeds: tuple[int, ...]) -> Mapping[int, TimeStep[ObservationT]]: ...

    def step(self, actions: Mapping[int, tuple[ActionT, ActionT]]) -> Mapping[int, TimeStep[ObservationT]]: ...

    def close(self) -> None: ...


class SlotGameBackend(Protocol[ObservationT, ActionT]):
    """Reset or step selected game slots without changing unselected slots."""

    def reset_slots(self, seeds: Mapping[int, int]) -> Mapping[int, TimeStep[ObservationT]]: ...

    def step(self, actions: Mapping[int, tuple[ActionT, ActionT]]) -> Mapping[int, TimeStep[ObservationT]]: ...

    def close(self) -> None: ...

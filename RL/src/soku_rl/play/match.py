"""Track original game rounds and complete matches without resetting the engine."""
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


@dataclass(frozen=True)
class MatchFrame:
    match: MatchState
    observations: tuple


@dataclass(frozen=True)
class MatchEvent:
    kind: str
    match: int
    round: int
    scores: tuple[int, int]


class MatchLifecycle:
    """Consume ordered snapshots. Scores decide results; zero HP only stops input.

    `round_started` tells the caller to reset observation history and policy memory.
    `match_finished` is emitted once when the configured number of wins is reached.
    Leaving battle without that result emits `match_interrupted`, not a loss.
    This observer does not control the game's rules, menus, or network connection.
    """

    def __init__(self, wins_required):
        if type(wins_required) is not int or wins_required < 1:
            raise ValueError("wins_required must be a positive integer")
        self.wins_required = wins_required
        self.phase = "waiting"
        self._match = 0
        self._round = -1
        self._frame = 0
        self._scores = (0, 0)
        self._active = False
        self._finished = False

    @property
    def can_act(self):
        return self.phase == "battle"

    def update(self, snapshot):
        if not isinstance(snapshot, MatchState):
            raise TypeError("match lifecycle requires a MatchState")
        in_battle = snapshot.phase == "battle"
        events = []

        def emit(kind):
            events.append(MatchEvent(kind, self._match, self._round, self._scores))

        if in_battle and snapshot.match != self._match:
            if snapshot.match < self._match:
                raise ValueError("match counter moved backwards")
            if self._active and not self._finished:
                emit("match_interrupted")
            self._match, self._round = snapshot.match, -1
            self._frame, self._scores = 0, (0, 0)
            self._active, self._finished = True, False
            emit("match_started")

        if self._active and snapshot.match == self._match:
            if snapshot.frame < self._frame:
                raise ValueError("simulation frame counter moved backwards")
            if any(new < old for new, old in zip(snapshot.scores, self._scores)):
                raise ValueError("round score decreased within a match")
            self._frame = snapshot.frame
            if snapshot.scores != self._scores:
                self._scores = snapshot.scores
                emit("score_changed")
            if not self._finished and max(self._scores) >= self.wins_required:
                self._finished = True
                emit("match_finished")

        if self._active and not in_battle:
            if not self._finished:
                emit("match_interrupted")
            self._active = False
        if not in_battle:
            self.phase = snapshot.phase
        elif self._finished:
            self.phase = "match_finished"
        elif min(snapshot.hp) <= 0:
            self.phase = "between_rounds"
        else:
            if snapshot.round != self._round:
                if snapshot.round < self._round:
                    raise ValueError("round counter moved backwards")
                self._round = snapshot.round
                emit("round_started")
            self.phase = "battle"
        return tuple(events)

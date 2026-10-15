"""Raw-state multi-instance API, independent of policies and learning rewards."""
from dataclasses import dataclass
from enum import Enum

from .batch import EngineBatch, RESET_METHODS
from .bridge_shared import RawFrameState
from .match import MatchConfig


class NativeOutcome(str, Enum):
    ONGOING = 'ongoing'
    P1_WIN = 'p1_win'
    P2_WIN = 'p2_win'
    DRAW = 'double_ko'


@dataclass(frozen=True)
class NativeFrame:
    raw: RawFrameState

    @property
    def frame(self):
        return int(self.raw.frameId)

    @property
    def outcome(self):
        left, right = self.raw.p1.hp <= 0, self.raw.p2.hp <= 0
        if left and right:
            return NativeOutcome.DRAW
        if left:
            return NativeOutcome.P2_WIN
        if right:
            return NativeOutcome.P1_WIN
        return NativeOutcome.ONGOING

    @property
    def ended(self):
        return self.outcome != NativeOutcome.ONGOING


@dataclass(frozen=True)
class NativeInput:
    inputs: tuple[int, int, int, int, int, int, int, int]


class RawReader:
    def read(self, raw, client, *, known_dropped=None):
        if raw.sceneId != 5 or raw.battleMode != 3:
            raise RuntimeError('game left VS battle')
        return NativeFrame(RawFrameState.from_buffer_copy(raw))

    def close(self):
        pass


class NativeGameBatch(EngineBatch):
    """Use original launch/reset/step ordering, return owned raw native states."""
    def enable_recording(self):
        raise NotImplementedError('raw recording needs an explicit replay adapter')

    def configure(self, match, *, max_frames=7200, mode='state'):
        if self.processes or mode not in RESET_METHODS or mode == 'image':
            raise ValueError('configure a supported raw-state mode before launch')
        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError('positive frame limit required')
        self.match = match if isinstance(match, MatchConfig) else MatchConfig(**match)
        self._check_match(self.match)
        self.observation_mode, self.max_frames = mode, max_frames

    def _create_reader(self, pid):
        return RawReader()

    def _create_recording(self, pid, slot, seed):
        raise NotImplementedError('a replay recording adapter must be supplied explicitly')

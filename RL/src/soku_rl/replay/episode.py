"""Bind an environment episode's exact end to its original replay bytes."""
from dataclasses import dataclass
import hashlib
import json

from soku_rl.pomg import Outcome


@dataclass(frozen=True)
class RecordedEpisode:
    frames: int
    reason: str

    def __post_init__(self):
        if type(self.frames) is not int or self.frames < 0:
            raise ValueError("recorded episode frames must be a nonnegative integer")
        if self.reason not in {"reset", "close", "time_limit", "p1_win", "p2_win", "double_ko"}:
            raise ValueError("unsupported recorded episode end reason")

    @classmethod
    def read(cls, source, replay, match_index):
        if match_index != 0 or len(replay.matches) != 1:
            raise ValueError("an environment episode must contain exactly one replay match")
        metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        if "scope" in metadata and metadata["scope"] != "episode":
            raise ValueError("a full match is not one environment episode; use input_stream conversion")
        if metadata["sha256"] != hashlib.sha256(replay.encode()).hexdigest():
            raise ValueError("episode metadata does not match the replay bytes")
        if type(metadata["seed"]) is not int or metadata["seed"] != replay.matches[0].seed:
            raise ValueError("episode metadata seed differs from the replay")
        return cls(metadata["frames"], metadata["reason"])

    def finished(self, state):
        if state.frame > self.frames or (state.ended and state.frame < self.frames):
            raise RuntimeError("replay ended at a different frame from the recorded episode")
        if state.frame != self.frames:
            return False
        expected = (self.reason if self.reason in {"p1_win", "p2_win", "double_ko"}
                    else Outcome.ONGOING.value)
        if state.outcome.value != expected:
            raise RuntimeError("replay outcome differs from the recorded episode")
        return True

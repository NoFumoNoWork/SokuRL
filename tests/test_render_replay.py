"""Check saved-trial identity and complete writes to an encoder pipe."""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from render_replay import check_trial_identity, write_pixels
from soku_rl.evaluation import make_plan


@dataclass
class StrategyIdentity:
    name: str
    fingerprint: str


def test_replay_identity_matches_both_seats_and_rejects_changed_game():
    strategies = {n: StrategyIdentity(n, f"{n}-weights") for n in ("a", "b")}
    for trial in make_plan(strategies, [1732], 91, "pinned-game"):
        check_trial_identity(asdict(trial), "pinned-game", 91)
        with pytest.raises(ValueError, match="artifacts"):
            check_trial_identity(asdict(trial), "changed-game", 91)


def test_encoder_writes_all_bytes_even_when_pipe_writes_are_partial():
    class PartialPipe:
        def __init__(self):
            self.received = bytearray()

        def write(self, values):
            part = values[:3]
            self.received.extend(part)
            return len(part)

    stream = PartialPipe()
    write_pixels(stream, bytes(range(24)))
    assert stream.received == bytes(range(24))

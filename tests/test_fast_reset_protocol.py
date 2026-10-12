import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import bridge_shared as b


def test_fast_reset_and_save_are_explicit_commands():
    client = b.BridgeClient.__new__(b.BridgeClient)
    calls = []
    client._send = lambda command,**kwargs: calls.append((command,kwargs)) or 7
    assert client.fast_reset() == 7
    assert client.save_branch_checkpoint() == 7
    assert calls == [(14,{}),(15,{})]


def test_fast_reset_cannot_relabel_seed():
    client = b.BridgeClient.__new__(b.BridgeClient)
    with pytest.raises(TypeError):
        client.fast_reset(seed=1732)


def test_snapshot_does_not_mix_old_frame_with_new_ack():
    class RacingBlock:
        def __init__(self):
            self.control = b.ControlBlock()
            self.control.currentFrame = 10
            self.control.latest.frameId = 10
            self.control.ackSeq = 1
            self.control.commandSeq = 2
            self.control.lastVerifiedFrame = b.NO_FRAME
            self.control.firstDivergentFrame = b.NO_FRAME
            self.reads = 0
        def __getattr__(self,name):
            return getattr(self.control,name)
        @property
        def statusSeq(self):
            self.reads += 1
            value = self.control.statusSeq
            if self.reads == 2:
                # Producer publishes a restore immediately AFTER the final
                # sequence read. The accepted older snapshot must retain ack1.
                self.control.statusSeq = 2
                self.control.currentFrame = 1
                self.control.latest.frameId = 1
                self.control.ackSeq = 2
            return value
    class Client(b.BridgeClient):
        @property
        def block(self): return self.fake
    client = Client.__new__(Client)
    client.fake = RacingBlock()
    snapshot = client.snapshot()
    assert snapshot.game_frame == snapshot.latest.frameId == 10
    assert snapshot.ack_seq == 1

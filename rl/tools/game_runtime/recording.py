"""Collect original replay records when an owned environment episode ends."""
from soku_rl.replay.recording import OriginalReplayReader
from game_runtime.privileged import ProcessMemory


class EpisodeRecording:
    def __init__(self, pid, slot, seed):
        self.memory = ProcessMemory(pid)
        self.reader = OriginalReplayReader(self.memory)
        self.slot, self.seed = slot, seed

    def finish(self, client, reason):
        try:
            replay = self.reader.read(client)
            snapshot = client.snapshot()
            if replay.matches[0].seed != self.seed:
                raise RuntimeError("recorded replay seed differs from the environment episode")
            return {"slot": self.slot, "seed": self.seed, "frames": snapshot.game_frame,
                    "reason": reason, "data": replay.encode()}
        finally:
            self.memory.close()

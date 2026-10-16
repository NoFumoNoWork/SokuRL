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
            if not client.original_replay_recording_enabled:
                raise RuntimeError("native REP export requires input-boundary protection")
            replay = self.reader.read(client)
            snapshot = client.snapshot()
            if replay.matches[0].seed != self.seed:
                raise RuntimeError("recorded replay seed differs from the environment episode")
            return {"slot": self.slot, "seed": self.seed, "frames": snapshot.game_frame,
                    "reason": reason, "data": replay.encode(),
                    "input_boundary": "NEUTRAL_UNTIL_NATIVE_RECORDING_READY"}
        finally:
            self.memory.close()

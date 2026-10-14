"""Read every paused offline frame from the existing complete native capture."""
import os

from game_runtime.privileged import PrivilegedReader
from game_runtime.snapshot_memory import SnapshotMemory


def offline_snapshot_requested():
    if 'SOKURL_OFFLINE_SNAPSHOT' not in os.environ:
        return False
    raise NotImplementedError("ABI12 Platform has no fork offline-snapshot sidecar; use paused process reads")


class OfflineSnapshotReader:
    def __init__(self, history):
        self.history = history
        self.cursor = 0
        self.reader = PrivilegedReader(SnapshotMemory([], b''))

    def observe(self, raw, client):
        before = client.snapshot()
        if before.run_state_name != 'PAUSED' or before.game_frame != raw.frameId:
            raise RuntimeError('offline snapshots require the exact paused frame')
        cursor, frames = self.history.read_after(self.cursor)
        if len(frames) != 1 or cursor != (self.cursor + 1) & 0xFFFFFFFF:
            raise RuntimeError('offline snapshots require exactly one new frame; no dropping or replay')
        capture = frames[0]
        if bytes(capture.raw) != bytes(raw):
            raise RuntimeError('offline snapshot differs from the requested native frame')
        self.reader.memory = capture.memory
        observations = self.reader.observe(raw, client)
        self.cursor = cursor
        return observations

    def close(self):
        self.history.close()
        self.reader.close()

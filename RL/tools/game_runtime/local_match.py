"""Own one paused local match, with complete observations and selected-seat control."""
from contextlib import ExitStack
import struct
import time

from bridge_shared import BridgeClient, wait_for_steps
from game_runtime.frames import wait_for_frame_zero
from game_runtime.observation import ObservationReader
from game_runtime.privileged import ProcessMemory
from soku_rl.env.observation.memory_schema import FIGHTER_FIELDS
from soku_rl.play.match import MatchFrame, MatchState
from soku_rl.replay.recording import OriginalReplayReader
import sokurl


class LocalMatch:
    def __init__(self, episode, seed, launch_timeout):
        if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF or launch_timeout <= 0:
            raise ValueError("local match requires a supported uint32 seed and positive timeout")
        self.resources = ExitStack()
        self.launch_timeout = launch_timeout
        self.seed = seed
        try:
            processes = sokurl._launch_vs_group_from_title(
                1, launch_timeout, headless=False, unlimited=False, seeds=(seed,),
                pause_at_start=True, capture_images=episode.observation_mode == "image",
                capture_state=episode.observation_mode == "state", match=episode.match)
            self.process = processes[0]
            self.resources.callback(self._close_process)
            self.client = BridgeClient(self.process.pid)
            self.resources.callback(self.client.close)
            wait_for_frame_zero(self.client, self.process.pid, launch_timeout)
            self.reader = ObservationReader(self.process.pid, episode.observation_mode, episode.visibility)
            self.resources.callback(self.reader.close)
            self.memory = ProcessMemory(self.process.pid)
            self.resources.callback(self.memory.close)
        except BaseException:
            self.resources.close()
            raise

    def _close_process(self):
        if self.process.is_running():
            sokurl.shutdown(5., self.process.pid)

    def _check_process(self):
        code = self.process.poll()
        if code is None:
            return
        if code != 0:
            raise RuntimeError(f"the owned local game failed with exit code {code}")
        raise EOFError("the owned local game closed")

    def read(self):
        self._check_process()
        before = self.client.snapshot()
        raw = before.latest
        if not before.in_gameplay or before.run_state_name != "PAUSED" or before.game_frame != raw.frameId:
            raise RuntimeError("local match observations require the exact paused battle frame")
        state = self.reader.read(raw, self.client)
        self.memory.begin_frame()
        battle, = struct.unpack("<I", self.memory.read(0x8985E4, 4))
        players = struct.unpack("<2I", self.memory.read(battle + 0xC, 8))
        offset, kind = FIGHTER_FIELDS["win_count"]
        scores = tuple(struct.unpack("<" + kind, self.memory.read(player + offset, 1))[0] for player in players)
        after = self.client.snapshot()
        if (after.run_state_name != "PAUSED" or after.game_frame != before.game_frame
                or after.latest.segmentId != raw.segmentId):
            raise RuntimeError("game advanced while reading local match scores")
        match = MatchState(raw.segmentId + 1, raw.roundId, state.frame, scores,
                           (raw.p1.hp, raw.p2.hp), "battle")
        return MatchFrame(match, state.observations)

    def step(self, inputs):
        self._check_process()
        frame = self.client.snapshot().game_frame
        sequence = self.client.step_controlled(inputs)
        deadline = time.monotonic() + 10.
        while True:
            self._check_process()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("local game did not complete the requested frame")
            try:
                wait_for_steps([self.client], [sequence], [frame + 1], min(.05, remaining))
                break
            except TimeoutError:
                # Recheck the same process and command; never submit a second
                # action when a short observation interval expires.
                continue
        self.client.drain_frames()

    def replay(self):
        replay = OriginalReplayReader(self.memory).read(self.client)
        if replay.matches[0].seed != self.seed:
            raise RuntimeError("local match replay seed differs from its initial state")
        return replay

    def reset(self, seed):
        segment = (self.client.snapshot().latest.segmentId + 1) & 0xFFFFFFFF
        sequence = self.client.reset_episode(seed)
        self.client.wait_for_reset(sequence, segment, seed, self.launch_timeout)
        self.seed = seed

    def close(self):
        self.resources.close()

"""Learning observation and recording adapters over Platform scheduling."""
import os
from soku_rl.env.observation.visibility import VisibilityConfig
from sokurl_platform.match import MatchConfig
from sokurl_platform.batch import EngineBatch, RESET_METHODS
from game_runtime.observation import ObservationReader
from game_runtime.offline_snapshot import offline_snapshot_requested


class SokuGameBatch(EngineBatch):
    def configure_observation(self, configuration):
        mode = configuration["mode"]
        if os.environ.get('SOKURL_NATIVE_INPUT_COUNTERS')=='1' and mode!='animation':
            raise NotImplementedError('counter adapter capability check currently requires animation mode')
        if self.fresh_reset and mode == "image":
            raise NotImplementedError("RGB parity is outside the verified fresh-reset scope")
        if self.processes or mode not in RESET_METHODS:
            raise ValueError("set a supported observation mode before launching games")
        if offline_snapshot_requested() and mode != "privileged_state":
            raise ValueError("offline snapshots require complete privileged observations")
        self.observation_mode = mode
        self.visibility = VisibilityConfig(**configuration["visibility"])
        self.match = MatchConfig(**configuration["match"])
        self._check_match(self.match)
        self.max_frames = configuration["max_frames"]
        self.animation_slots = configuration.get('animation_object_slots', 16)
        self.passive_policy=configuration.get('passive_policy',{})
        if type(self.max_frames) is not int or self.max_frames < 1:
            raise ValueError("a positive episode frame limit is required")

    def _create_reader(self, pid):
        return ObservationReader(pid, self.observation_mode, self.visibility,
                                 self.animation_slots, self.max_frames, self.passive_policy)

    def _create_recording(self, pid, slot, seed):
        from game_runtime.recording import EpisodeRecording
        return EpisodeRecording(pid, slot, seed)

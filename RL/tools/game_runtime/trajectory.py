"""Replay a selected original match through the environment observation code."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import struct
from tempfile import TemporaryDirectory

import numpy as np

from game_runtime.stepping import input_tuple
from game_runtime.observation import ObservationReader
from game_runtime.replay import launch_replay
from game_runtime.privileged import ProcessMemory
from soku_rl.env.encoding import AGENTS
from soku_rl.env.observation_history import ObservationHistory
from soku_rl.pomg import Outcome
from soku_rl.replay.rollout import RolloutFrame
from soku_rl.replay.episode import RecordedEpisode


def replay_frames(replay, match_index, config, launch_timeout, scratch_directory, boundary):
    if type(match_index) is not int or not 0 <= match_index < len(replay.matches):
        raise ValueError("a valid replay match index is required")
    if isinstance(boundary, RecordedEpisode):
        if config.max_frames < boundary.frames:
            raise ValueError("episode.max_frames cannot truncate a recorded episode")
    elif boundary != "input_stream":
        raise ValueError("replay boundary must be input_stream or a recorded episode")
    header = bytearray(replay.header)
    header[7] = 1
    selected = replace(replay, header=bytes(header), matches=(replay.matches[match_index],))
    with ExitStack() as resources:
        directory = resources.enter_context(TemporaryDirectory(prefix="sokurl-replay-", dir=scratch_directory))
        path = Path(directory) / "selected.rep"
        path.write_bytes(selected.encode())
        instance = launch_replay(path, launch_timeout, True, config.observation_mode)
        resources.callback(instance.close)
        reader = ObservationReader(instance.pid, config.observation_mode, config.visibility)
        resources.callback(reader.close)
        memory = ProcessMemory(instance.pid)
        resources.callback(memory.close)
        history = ObservationHistory(config, AGENTS)
        raw = instance.state
        while True:
            state = reader.read(raw, instance.client)
            if raw.frameId == 0:
                history.reset(state.frame, state.observations)
            else:
                history.append(state.frame, state.observations)
            memory.begin_frame()
            record, = struct.unpack("<I", memory.read(0x898718 + 0x104, 4))
            if not record:
                raise RuntimeError("original replay input record is unavailable")
            remaining, = struct.unpack("<I", memory.read(record + 0x4C, 4))
            ended = (boundary.finished(state) if isinstance(boundary, RecordedEpisode)
                     else state.ended or state.frame >= config.max_frames or remaining == 0)
            outcome = state.outcome
            if ended and not state.terminated:
                outcome = Outcome.TRUNCATED
            observations = history.observations()
            yield RolloutFrame(state.frame, np.stack([observations[a] for a in AGENTS]),
                np.asarray((input_tuple(raw.p1.input), input_tuple(raw.p2.input)), dtype=np.int32),
                state.rewards, outcome.value)
            if ended:
                return
            raw = instance.step_native(10.0)

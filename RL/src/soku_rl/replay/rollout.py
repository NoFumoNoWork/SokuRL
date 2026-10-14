"""Stream environment observations with their unchanged original replay."""
from dataclasses import asdict, dataclass
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np

from soku_rl.replay.format import Replay


@dataclass(frozen=True)
class RolloutFrame:
    frame: int
    observations: np.ndarray
    engine_inputs: np.ndarray
    rewards: tuple
    outcome: str


def write_rollout(path, replay, match_index, episode, frames):
    if not isinstance(replay, Replay) or not 0 <= match_index < len(replay.matches):
        raise ValueError("a decoded replay and a valid match index are required")
    count = 0
    expected_shape = (2, *episode.space().shape)
    with ZipFile(path, "x") as archive:
        archive.writestr("original.rep", replay.encode())
        for frame in frames:
            if frame.frame != count:
                raise ValueError("rollout frames must start at zero and remain consecutive")
            if frame.observations.shape != expected_shape:
                raise ValueError("rollout observations differ from the environment space")
            if frame.engine_inputs.shape != (2, 8):
                raise ValueError("rollout requires both players' eight engine input counters")
            data = BytesIO()
            np.savez_compressed(data, observations=frame.observations,
                                engine_inputs=frame.engine_inputs,
                                rewards=np.asarray(frame.rewards, dtype=np.float64),
                                outcome=np.asarray(frame.outcome))
            archive.writestr(f"frames/{count:08d}.npz", data.getvalue())
            count += 1
        if count == 0:
            raise ValueError("a rollout must include its initial frame")
        archive.writestr("manifest.json", json.dumps({
            "schema": 1, "frames": count, "match_index": match_index,
            "episode": asdict(episode),
            "inputs": "engine counters after original game input processing",
        }, indent=2))
    return count


def read_manifest(archive):
    manifest = json.loads(archive.read("manifest.json"))
    if manifest["schema"] != 1 or type(manifest["frames"]) is not int or manifest["frames"] < 1:
        raise ValueError("unsupported or incomplete rollout archive")
    return manifest


def read_rollout(path):
    with ZipFile(path) as archive:
        manifest = read_manifest(archive)
        for frame in range(manifest["frames"]):
            with np.load(BytesIO(archive.read(f"frames/{frame:08d}.npz")), allow_pickle=False) as record:
                yield RolloutFrame(frame, record["observations"], record["engine_inputs"],
                                   tuple(record["rewards"]), str(record["outcome"]))


def export_replay(rollout, destination):
    with ZipFile(rollout) as archive:
        read_manifest(archive)
        data = archive.read("original.rep")
        if Replay.decode(data).encode() != data:
            raise ValueError("archived replay is not lossless")
    with destination.open("xb") as target:
        target.write(data)

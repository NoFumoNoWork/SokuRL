"""Replay saved decisions through the real engine and encode its RGB frames."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import subprocess
import time

import hydra
import numpy as np
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.env.control import ControlConfig, DelayedControls
from soku_rl.env.encoding import AGENTS
from soku_rl.learning_wrappers import LearningConfig, LearningInterface
from soku_rl.worker_pipe import WorkerBackend


def write_pixels(stream, pixels):
    remaining = memoryview(pixels)
    while remaining:
        count = stream.write(remaining)
        if not count:
            raise BrokenPipeError("video encoder stopped accepting frames")
        remaining = remaining[count:]


def check_trial_identity(record, game_id, policy_seed):
    players = record["players"][::-1] if record["swapped"] else record["players"]
    identities = record["strategy_ids"][::-1] if record["swapped"] else record["strategy_ids"]
    data = [game_id, record["world_seed"], *players, identities, policy_seed]
    block_id = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    if block_id != record["block_id"]:
        raise ValueError("replay game artifacts or trial identity differ from the recorded block")


@hydra.main(version_base="1.3", config_path="../config", config_name="replay")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    replay = config["replay"]
    source = Path(replay["evaluation_directory"]).resolve(strict=True)
    # progress.json is also retained by completed benchmarks.
    progress_bytes = (source / "progress.json").read_bytes()
    records = json.loads(progress_bytes)["games"]
    matches = [record for record in records if record["trial_id"] == replay["trial_id"]]
    if len(matches) != 1 or matches[0]["status"] != "complete":
        raise ValueError("replay requires exactly one completed benchmark trial")
    record = matches[0]
    source_config = OmegaConf.to_container(OmegaConf.load(source / "config.yaml"), resolve=True)
    episode = EpisodeConfig(**source_config["episode"])
    interface = LearningInterface(episode, LearningConfig(**source_config["wrappers"]))
    trace = (source / record["replay"]).resolve(strict=True)
    if trace.parent != source:
        raise ValueError("replay trace must be in the evaluation directory")
    with np.load(trace, allow_pickle=False) as saved:
        seed, actions = int(saved["seed"]), saved["actions"].copy()
    if (seed != record["world_seed"] or actions.ndim != 2 or actions.shape[1] != 2
            or not np.issubdtype(actions.dtype, np.integer) or (actions < 0).any()
            or (actions >= interface.action_space.n).any()):
        raise ValueError("invalid recorded decisions or world seed")
    expected_decisions = (record["frames"] + episode.decision_frames - 1) // episode.decision_frames
    if len(actions) != expected_decisions or not 0 < record["frames"] <= episode.max_frames:
        raise ValueError("recorded decision count and simulation length disagree")
    if type(replay["encoder_gpu"]) is not int or replay["encoder_gpu"] < 0 or not replay["ffmpeg"]:
        raise ValueError("an explicit NVENC GPU and ffmpeg command are required")
    directory = Path(config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    (directory / "source-config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(source_config)), encoding="utf-8")
    (directory / "source-record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    command = [*replay["ffmpeg"], "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", "320x240", "-framerate", "60", "-i", "pipe:0",
               "-an", "-c:v", "h264_nvenc", "-gpu", str(replay["encoder_gpu"]), "-preset", replay["encoder_preset"],
               "-cq", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(directory / "replay.mp4")]
    report = {"success": False, "source_record": record,
              "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
              "source_progress_sha256": hashlib.sha256(progress_bytes).hexdigest(),
              "encoder_command": command, "fps": 60, "pixel_frames": 0}
    started = time.perf_counter()
    hashes = []
    try:
        with closing(WorkerBackend(log_path=directory / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation({"mode": "image", "visibility": source_config["episode"]["visibility"]})
            report["runtime"] = backend.identity
            # The trial block includes the original artifact hash, seed and both
            # policy identities. Check it even while the benchmark is running.
            check_trial_identity(record, backend.identity["fingerprints"]["game_id"],
                                 source_config["benchmark"]["policy_seed"])
            state = backend.reset_slots({0: seed})[0]
            if state.frame != 0 or state.ended:
                raise RuntimeError("replay did not start from an ongoing frame zero")
            controls = DelayedControls(ControlConfig(episode.decision_frames, episode.latency_frames))
            with (directory / "encoder.log").open("wb") as log:
                encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                try:
                    write_pixels(encoder.stdin, state.observations[0].pixels)
                    report["pixel_frames"] += 1
                    hashes.append(state.diagnostics["hash"])
                    for index, joint in enumerate(actions):
                        if state.ended or state.frame != index * episode.decision_frames:
                            raise RuntimeError("replay terminated before its recorded decisions ended")
                        commands = {a: interface.command(joint[i]) for i, a in enumerate(AGENTS)}
                        controls.submit(state.frame, commands)
                        for _ in range(episode.decision_frames):
                            state = backend.step({0: controls.inputs(state.frame)})[0]
                            if state.observations[0].frame != state.frame:
                                raise RuntimeError("video pixels and simulation frame disagree")
                            write_pixels(encoder.stdin, state.observations[0].pixels)
                            report["pixel_frames"] += 1
                            hashes.append(state.diagnostics["hash"])
                            if state.ended or state.frame >= episode.max_frames:
                                break
                    outcome = state.outcome.value if state.ended else "time_limit"
                    report["actual"] = {"frames": state.frame, "outcome": outcome, "final": state.diagnostics}
                    if state.frame != record["frames"] or outcome != record["outcome"]:
                        raise RuntimeError("rendered replay differs from the recorded game result")
                finally:
                    encoder.stdin.close()
                    status = encoder.wait(timeout=60)
                    if status:
                        raise RuntimeError(f"video encoder exited with code {status}; see encoder.log")
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (directory / "state-hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
        (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

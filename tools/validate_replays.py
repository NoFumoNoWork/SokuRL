"""Reproduce native reset failures using completed benchmark action records."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.env.control import ControlConfig, DelayedControls
from soku_rl.env.encoding import AGENTS
from soku_rl.learning_wrappers import LearningConfig, LearningInterface
from soku_rl.worker_pipe import WorkerBackend
from render_replay import check_trial_identity


@hydra.main(version_base="1.3", config_path="../config", config_name="validate_replays")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    validation = config["replay_validation"]
    source = Path(validation["evaluation_directory"]).resolve(strict=True)
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    records = {r["trial_id"]: r for r in json.loads((source / "progress.json").read_text())["games"]}
    selected = validation["trial_indices"]
    if (not selected or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < len(plan) for i in selected)):
        raise ValueError("distinct valid plan indices are required")
    if type(validation["cycles"]) is not int or validation["cycles"] < 1:
        raise ValueError("cycles must be a positive integer")
    if type(validation["prefix_frames"]) is not int or validation["prefix_frames"] < 0:
        raise ValueError("prefix_frames must be a nonnegative integer")
    if type(validation["next_seed"]) is not int or not 0 <= validation["next_seed"] < 0xFFFFFFFF:
        raise ValueError("next_seed must be a supported native seed")
    source_config = OmegaConf.to_container(OmegaConf.load(source / "config.yaml"), resolve=True)
    episode = EpisodeConfig(**source_config["episode"])
    interface = LearningInterface(episode, LearningConfig(**source_config["wrappers"]))
    trials, actions = {}, {}
    for slot, index in enumerate(selected):
        record = records[plan[index]["trial_id"]]
        if record["status"] != "complete":
            raise ValueError("only completed action traces can be replayed")
        path = (source / record["replay"]).resolve(strict=True)
        if path.parent != source:
            raise ValueError("trace must be inside the source evaluation directory")
        with np.load(path, allow_pickle=False) as saved:
            seed, value = int(saved["seed"]), saved["actions"].copy()
        decisions = (record["frames"] + episode.decision_frames - 1) // episode.decision_frames
        if (seed != record["world_seed"] or value.shape != (decisions, 2)
                or not np.issubdtype(value.dtype, np.integer) or (value < 0).any()
                or (value >= interface.action_space.n).any()):
            raise ValueError("invalid saved action trace")
        trials[slot], actions[slot] = record, value
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    (output / "source-config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(source_config)), encoding="utf-8")
    report = {"success": False, "cycles": [], "trials": trials}
    started = time.perf_counter()

    def save():
        report["seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        with closing(WorkerBackend(log_path=output / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(episode.backend_observation())
            report["runtime"] = backend.identity
            game_id = backend.identity["fingerprints"]["game_id"]
            if validation["bridge_artifact"] != "recorded":
                original = json.loads((source / "result.json").read_text())["runtime"]["fingerprints"]
                actual = backend.identity["fingerprints"]["artifact_hashes"]
                expected = original["artifact_hashes"] | {
                    "modules/SokuRLBridge/SokuRLBridge.dll": validation["bridge_artifact"]}
                if actual != expected:
                    raise ValueError("diagnostic replay may change only the explicitly pinned bridge DLL")
                report["diagnostic_bridge_artifact"] = validation["bridge_artifact"]
                game_id = original["game_id"]
            for trial in trials.values():
                check_trial_identity(trial, game_id,
                                     source_config["benchmark"]["policy_seed"])
            reference = {}
            pids = {}
            for cycle in range(validation["cycles"]):
                report["stage"] = {"cycle": cycle, "operation": "reset_before_replay"}
                save()
                states = backend.reset_slots({slot: trial["world_seed"] for slot, trial in trials.items()})
                controls = {slot: DelayedControls(ControlConfig(episode.decision_frames, episode.latency_frames))
                            for slot in trials}
                digests = {slot: hashlib.sha256() for slot in trials}
                for slot, state in states.items():
                    if cycle and state.diagnostics["pid"] != pids[slot]:
                        raise RuntimeError("replay reset replaced an engine process")
                    pids[slot] = state.diagnostics["pid"]
                    digests[slot].update(state.diagnostics["hash"].encode())
                current = {"cycle": cycle, "episodes": [], "pids": dict(pids)}
                report["stage"] = {"cycle": cycle, "operation": "replay"}
                save()
                while states:
                    for slot in list(states):
                        state = states[slot]
                        limit = min(trials[slot]["frames"], validation["prefix_frames"])
                        if state.ended or state.frame >= limit:
                            complete = limit == trials[slot]["frames"]
                            outcome = state.outcome.value if state.ended or not complete else "time_limit"
                            if state.frame != limit or (complete and outcome != trials[slot]["outcome"]):
                                raise RuntimeError(f"trial {selected[slot]} changed outcome or length: {state.frame}, {outcome}")
                            digest = digests[slot].hexdigest()
                            if cycle and digest != reference[slot]:
                                raise RuntimeError(f"trial {selected[slot]} changed its native state trajectory")
                            reference[slot] = digest
                            current["episodes"].append({"slot": slot, "frames": state.frame, "outcome": outcome,
                                "complete_replay": complete, "state_trajectory_sha256": digest,
                                "final": dict(state.diagnostics)})
                            del states[slot]
                    if not states:
                        break
                    for slot, state in states.items():
                        if state.frame % episode.decision_frames == 0:
                            decision = actions[slot][state.frame // episode.decision_frames]
                            controls[slot].submit(state.frame,
                                {agent: interface.command(decision[i]) for i, agent in enumerate(AGENTS)})
                    states = backend.step({slot: controls[slot].inputs(state.frame) for slot, state in states.items()})
                    for slot, state in states.items():
                        digests[slot].update(state.diagnostics["hash"].encode())
                report["cycles"].append(current)
                report["stage"] = {"cycle": cycle, "operation": "reset_after_replay"}
                save()
                states = backend.reset_slots(dict.fromkeys(trials, validation["next_seed"]))
                for slot, state in states.items():
                    if state.frame != 0 or state.ended or state.diagnostics["pid"] != pids[slot]:
                        raise RuntimeError("reset after replay did not retain a clean engine instance")
                current["next_episode"] = {slot: dict(state.diagnostics) for slot, state in states.items()}
                save()
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

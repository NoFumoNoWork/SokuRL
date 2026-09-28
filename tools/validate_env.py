"""Validate complete real-game episodes through the configured worker transport."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
from soku_rl.env.encoding import AGENTS
from soku_rl.worker_pipe import WorkerBackend


@hydra.main(version_base="1.3", config_path="../config", config_name="validate")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    episode = EpisodeConfig(**config["episode"])
    count = config["validation"]["episodes_per_slot"]
    if type(count) is not int or count < 1:
        raise ValueError("episodes_per_slot must be a positive integer")
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    report = {"success": False, "episodes": []}
    started = time.perf_counter()
    try:
        with closing(WorkerBackend(log_path=output / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(episode.observation_mode)
            report["runtime"] = backend.identity
            env = TwoPlayerVectorEnv(backend, config["num_envs"], episode)
            rng = np.random.default_rng(config["seed"])
            completed = dict.fromkeys(range(env.num_envs), 0)
            seeds = {s: int(rng.integers(0, 0xFFFFFFFF)) for s in completed}
            observations, infos = env.reset(seeds)
            frames = dict.fromkeys(completed, 0)
            traces = {s: [] for s in completed}
            while observations:
                for players in observations.values():
                    for agent in AGENTS:
                        if not env.single_observation_space.contains(players[agent]):
                            raise RuntimeError("observation violates its declared space")
                actions = {s: {a: int(rng.integers(env.single_action_space.n)) for a in AGENTS}
                           for s in observations}
                for slot, joint in actions.items():
                    traces[slot].append([joint[a] for a in AGENTS])
                observations, rewards, terms, truncs, infos = env.step(actions)
                resets = {}
                for slot in list(observations):
                    info = infos[slot][AGENTS[0]]
                    advanced = info["frame"] - frames[slot]
                    if not 1 <= advanced <= episode.decision_frames:
                        raise RuntimeError("invalid simulation frame advance")
                    if sum(rewards[slot].values()) != 0:
                        raise RuntimeError("game rewards are not zero sum")
                    frames[slot] = info["frame"]
                    done = terms[slot][AGENTS[0]] or truncs[slot][AGENTS[0]]
                    if not done:
                        continue
                    completed[slot] += 1
                    name = f"slot-{slot}-episode-{completed[slot]}"
                    np.savez_compressed(output / (name + ".npz"), seed=seeds[slot],
                                        actions=np.asarray(traces[slot], dtype=np.int16))
                    report["episodes"].append({"slot": slot, "seed": seeds[slot],
                        "decisions": len(traces[slot]), "frames": frames[slot],
                        "outcome": info["outcome"], "trace": name + ".npz",
                        "final_observation_sha256": hashlib.sha256(
                            observations[slot][AGENTS[0]].tobytes()).hexdigest()})
                    if completed[slot] < count:
                        resets[slot] = int(rng.integers(0, 0xFFFFFFFF))
                    del observations[slot]
                if resets:
                    fresh, _ = env.reset(resets)
                    observations.update(fresh)
                    seeds.update(resets)
                    for slot in resets:
                        frames[slot], traces[slot] = 0, []
                if resets or not observations:
                    (output / "progress.json").write_text(json.dumps(report["episodes"], indent=2), encoding="utf-8")
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

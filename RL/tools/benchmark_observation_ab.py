"""Interleaved resident-instance A/B with an isolated diagnostic worker switch."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time

import numpy as np


def diagnostic_worker(initial_mode=None):
    from game_runtime.batch import SokuGameBatch
    from game_runtime import observation
    from soku_rl.env.observation import visible_state
    optimized_encode = observation.observe_visible_states
    optimized_read = observation.ObservationReader.read
    original_status = SokuGameBatch.status

    def baseline_encode(raw, render, config):
        poses, objects = visible_state.visible_entities(render, config)
        return tuple(visible_state._encode(raw, poses, objects, p, config) for p in (0, 1))

    def baseline_read(self, raw, bridge, *, known_dropped=None):
        return optimized_read(self, raw, bridge)

    def set_mode(mode):
        if mode not in ("baseline", "optimized"):
            raise ValueError("invalid audit mode")
        observation.observe_visible_states = baseline_encode if mode == "baseline" else optimized_encode
        observation.ObservationReader.read = baseline_read if mode == "baseline" else optimized_read

    def status(self, request):
        if isinstance(request, dict):
            set_mode(request["audit_mode"])
            request = request["slots"]
        return original_status(self, request)

    SokuGameBatch.status = status
    if initial_mode is not None:
        set_mode(initial_mode)
    runpy.run_path("tools/rollout_worker.py", run_name="__main__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--decisions", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(os.environ["SOKURL_PLATFORM_ROOT"])
    os.chdir(root)
    sys.path[:0] = [str(root / "src"), str(root / "tools")]
    if args.worker:
        diagnostic_worker()
        return
    if args.output is None or args.decisions < 8 or args.pairs < 2:
        parser.error("output, decisions>=8 and pairs>=2 required")
    from omegaconf import OmegaConf
    from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
    from soku_rl.env.encoding import AGENTS, encode_action
    from soku_rl.env.worker_pipe import WorkerBackend
    from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
    config = OmegaConf.to_container(OmegaConf.load("logs/training/platform-initial-4-20261013-124845/config.yaml"), resolve=True)
    report = {"status": "RUNNING", "scope": "full environment pipeline including IPC, excluding learner",
              "trials": [], "workers": 4, "decisions": args.decisions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    backend = WorkerBackend([sys.executable, "-B", "tools/benchmark_observation_ab.py", "--worker"],
        str(root), str(args.output.with_suffix(".worker.log")), 120, 40, True,
        str(root / "th123_jp"), fresh_reset=True, cards_disabled=True, fixed_weather="clear")
    env = None
    try:
        references = {}
        expected_pids = None
        for latency in (0, 5):
            episode = replace(EpisodeConfig.from_dict(config["episode"]),
                              latency_frames=latency, max_frames=args.decisions * 3 + 10)
            if env is None:
                backend.configure_observation(episode.backend_observation())
            env = LearningVectorEnv(TwoPlayerVectorEnv(backend, 4, episode), LearningConfig(**config["wrappers"]))
            for pair in range(-1, args.pairs):
                order = ("baseline", "optimized") if pair % 2 == 0 else ("optimized", "baseline")
                for mode in order:
                    seeds = {s: 1732 + s for s in range(4)}
                    env.reset(seeds)
                    statuses = backend.request("status", {"audit_mode": mode, "slots": list(seeds)})
                    pids = {s: status["pid"] for s, status in statuses.items()}
                    if expected_pids is None:
                        expected_pids = pids
                    assert expected_pids == pids, "resident game PID changed"
                    digest = hashlib.sha256()
                    elapsed = 0.
                    for decision in range(args.decisions):
                        controls = ((-1 if decision % 80 < 40 else 1, 0, int(decision % 30 < 2),
                                     int(decision % 50 < 2 and not decision % 30 < 2), 0, 0, 0, 0),
                                    (1 if decision % 80 < 40 else -1, 0, int(decision % 40 < 2),
                                     0, int(decision % 60 < 2 and not decision % 40 < 2), 0, 0, 0))
                        actions = {s: {a: env.interface.action(encode_action(controls[i]))
                                       for i, a in enumerate(AGENTS)} for s in seeds}
                        started = time.perf_counter()
                        obs, rewards, terms, truncs, infos = env.step(actions)
                        elapsed += time.perf_counter() - started
                        for s in seeds:
                            assert not any(terms[s].values()) and not any(truncs[s].values()), "unexpected early terminal"
                            for a in AGENTS:
                                digest.update(obs[s][a].tobytes())
                            digest.update(json.dumps([rewards[s], terms[s], truncs[s]], sort_keys=True).encode())
                        # Validation reads are identical in both modes and excluded from the timed window.
                        statuses = backend.request("status", list(seeds))
                        digest.update(json.dumps([(s, statuses[s]["frame"], statuses[s]["hash"])
                                                  for s in seeds]).encode())
                    value = digest.hexdigest()
                    if latency not in references:
                        references[latency] = value
                    assert value == references[latency], "native/public/reward trajectory diverged"
                    if pair >= 0:
                        report["trials"].append({"latency": latency, "pair": pair, "mode": mode,
                            "seconds": elapsed, "trajectory_hash": value})
                        save()
            # Do not close the borrowed backend between configurations.
        report["summary"] = []
        for latency in (0, 5):
            rows = [r for r in report["trials"] if r["latency"] == latency]
            baseline = [r["seconds"] for r in rows if r["mode"] == "baseline"]
            optimized = [r["seconds"] for r in rows if r["mode"] == "optimized"]
            report["summary"].append({"latency": latency, "baseline_mean_seconds": statistics.mean(baseline),
                "optimized_mean_seconds": statistics.mean(optimized),
                "baseline_stdev": statistics.stdev(baseline), "optimized_stdev": statistics.stdev(optimized),
                "speedup": statistics.mean(baseline) / statistics.mean(optimized)})
        report.update(status="PASS", resident_pids=expected_pids, parity="PASS")
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        backend.close()
        save()
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

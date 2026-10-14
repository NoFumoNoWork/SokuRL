"""Opt-in boundary timers for environment interaction, without learner profiling."""
import argparse
from collections import defaultdict
from datetime import datetime
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time


class Timers:
    def __init__(self):
        self.rows = defaultdict(lambda: {"calls": 0, "inclusive_ns": 0, "exclusive_ns": 0,
                                         "max_ns": 0})
        self.local = threading.local()

    def wrap(self, owner, name, label, request=False):
        original = getattr(owner, name)

        @wraps(original)
        def timed(*args, **kwargs):
            key = label + (":" + str(args[1]) if request else "")
            stack = getattr(self.local, "stack", None)
            if stack is None:
                stack = self.local.stack = []
            frame = [time.perf_counter_ns(), 0]
            stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - frame[0]
                stack.pop()
                if stack:
                    stack[-1][1] += elapsed
                row = self.rows[key]
                row["calls"] += 1
                row["inclusive_ns"] += elapsed
                row["exclusive_ns"] += elapsed - frame[1]
                row["max_ns"] = max(row["max_ns"], elapsed)

        setattr(owner, name, timed)

    def install(self, worker):
        if worker:
            specs = [
                ("bridge_shared", "BridgeClient", "snapshot"),
                ("bridge_shared", "BridgeClient", "step_with_inputs"),
                ("bridge_shared", "BridgeClient", "wait_for_reset"),
                ("game_runtime.batch", None, "wait_for_steps"),
                ("game_runtime.batch", None, "drain_frames_into"),
                ("game_runtime.batch", "SokuGameBatch", "step"),
                ("game_runtime.batch", "SokuGameBatch", "reset_slots"),
                ("game_runtime.batch", "SokuGameBatch", "_launch_slots"),
                ("game_runtime.observation", "ObservationReader", "read"),
                ("game_runtime.observation", None, "time_step"),
                ("game_runtime.observation", None, "observe_visible_states"),
                ("game_runtime.images", "ImageClient", "read_state"),
                ("game_runtime.images", "ImageClient", "_read"),
                ("soku_rl.env.observation.visible_state", None, "visible_entities"),
                ("soku_rl.env.observation.visible_state", None, "_encode"),
                ("soku_rl.env.observation.visibility", None, "visible_fraction"),
                ("soku_rl.env.worker_pipe", None, "send"),
            ]
        else:
            specs = [
                ("soku_rl.env.worker_pipe", "WorkerConnection", "request"),
                ("soku_rl.env.vector_env", "TwoPlayerVectorEnv", "step"),
                ("soku_rl.env.hisouten_env", "Episode", "step"),
                ("soku_rl.env.hisouten_env", "Episode", "actions"),
                ("soku_rl.env.hisouten_env", "Episode", "_infos"),
                ("soku_rl.env.hisouten_env", "EpisodeConfig", "encode"),
                ("soku_rl.env.observation_history", "ObservationHistory", "observations"),
                ("soku_rl.env.wrappers.learning", "LearningVectorEnv", "step"),
                ("soku_rl.env.wrappers.learning", "LearningEpisode", "step"),
                ("soku_rl.env.wrappers.learning", "LearningEpisode", "observation"),
                ("soku_rl.rl.opponent_env", "OpponentMixtureVecEnv", "step_wait"),
                ("soku_rl.rl.opponent_env", "OpponentMixtureVecEnv", "_reset_slots"),
                ("soku_rl.policy.rules.observed_rules", "LearningRuleEpisode", "act"),
            ]
        for module, cls, method in specs:
            owner = importlib.import_module(module)
            if cls:
                owner = getattr(owner, cls)
            label = ".".join(x for x in (module, cls, method) if x)
            self.wrap(owner, method, label, request=method == "request")

    def save(self, path):
        rows = [{"label": key, "calls": row["calls"],
                 "inclusive_seconds": row["inclusive_ns"] / 1e9,
                 "exclusive_seconds": row["exclusive_ns"] / 1e9,
                 "max_seconds": row["max_ns"] / 1e9}
                for key, row in self.rows.items()]
        rows.sort(key=lambda x: x["inclusive_seconds"], reverse=True)
        path.write_text(json.dumps({"rows": rows,
            "scope": "selected environment boundaries only; no learner or per-instruction profiling",
            "exclusive": "excludes only nested selected boundaries, not all child computation",
            "overlap": "parent waits overlap worker; never sum across processes"}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    root = Path(os.environ["SOKURL_PLATFORM_ROOT"])
    os.chdir(root)
    sys.path[:0] = [str(root / "src"), str(root / "tools")]
    timers = Timers()
    timers.install(worker=args.worker_output is not None)
    if args.worker_output:
        try:
            runpy.run_path("tools/rollout_worker.py", run_name="__main__")
        finally:
            timers.save(args.worker_output)
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = root / "docs/reports/environment_interaction" / stamp
    output.mkdir(parents=True, exist_ok=False)
    training = root / "RL/runs/training" / ("platform-env-timing-4-" + stamp)
    command = json.dumps([sys.executable.replace("\\", "/"), "-B",
                          "tools/profile_environment_interaction.py", "--worker-output",
                          (output / "worker.json").as_posix()])
    sys.argv = ["tools/train.py", "--config-name", "train_platform_initial",
                "runtime.command=" + command, "output=" + training.as_posix()]
    (output / "run.json").write_text(json.dumps({"training": str(training),
        "arguments": sys.argv}, indent=2), encoding="utf-8")
    try:
        runpy.run_path("tools/train.py", run_name="__main__")
    finally:
        timers.save(output / "parent.json")
        print("Environment timing directory:", output)


if __name__ == "__main__":
    main()

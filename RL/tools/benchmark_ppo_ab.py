"""Repeat complete unprofiled PPO with diagnostic baseline/optimized code paths."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys


def baseline_vector_step(self, actions):
    self._check_slots(actions)
    for slot, action in actions.items():
        self.episodes[slot].actions(action)
    try:
        for slot, action in actions.items():
            self.episodes[slot].submit(action)
        results, active = {}, set(actions)
        for _ in range(next(iter(self.episodes.values())).config.decision_frames):
            joint = {s: self.episodes[s].inputs() for s in sorted(active)}
            states = self.backend.step(joint)
            if set(states) != active:
                raise RuntimeError("backend returned incorrect step slots")
            results.update({s: self.episodes[s].step(states[s]) for s in active})
            active = {s for s in active if not self.episodes[s].ended}
            if not active:
                break
    except BaseException:
        for slot in actions:
            self.episodes[slot].invalidate()
        raise
    return tuple({s: r[i] for s, r in results.items()} for i in range(5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-mode", choices=["baseline", "optimized"])
    parser.add_argument("--single-mode", choices=["baseline", "optimized"])
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--timesteps", type=int, default=8192)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    root = Path(os.environ["SOKURL_PLATFORM_ROOT"])
    os.chdir(root)
    sys.path[:0] = [str(root / "src"), str(root / "tools")]
    if args.worker_mode:
        from benchmark_observation_ab import diagnostic_worker
        diagnostic_worker(args.worker_mode)
        return
    if args.single_mode:
        if args.single_mode == "baseline":
            from soku_rl.env.vector_env import TwoPlayerVectorEnv
            TwoPlayerVectorEnv.step = baseline_vector_step
        command = json.dumps([sys.executable.replace("\\", "/"), "-B",
                              "tools/benchmark_ppo_ab.py", "--worker-mode", args.single_mode])
        sys.argv = ["tools/train.py", "--config-name", "train_platform_initial",
                    "runtime.command=" + command, "output=" + args.training_output.as_posix(),
                    "algorithm.timesteps=" + str(args.timesteps)]
        runpy.run_path("tools/train.py", run_name="__main__")
        return
    if args.pairs < 2 or args.timesteps < 2048 or args.timesteps % 2048:
        parser.error("pairs>=2 and timesteps a positive multiple of2048 required")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = root / "docs/reports/environment_interaction" / ("ppo-ab-" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "pairs": args.pairs, "timesteps": args.timesteps,
              "workers": 4, "custom_profilers": "OFF", "runs": []}
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    try:
        expected_hash = None
        for pair in range(args.pairs):
            for mode in (("baseline", "optimized") if pair % 2 == 0 else ("optimized", "baseline")):
                training = root / "RL/runs/training" / f"ppo-ab-{stamp}-{pair}-{mode}"
                command = [sys.executable, "-B", "tools/benchmark_ppo_ab.py", "--single-mode", mode,
                           "--training-output", str(training), "--timesteps", str(args.timesteps)]
                with (output / f"{pair}-{mode}.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=root)
                if completed.returncode:
                    raise RuntimeError(f"{mode} failed, see {pair}-{mode}.log")
                result = json.loads((training / "result.json").read_text())
                if result["success"] is not True:
                    raise RuntimeError("training reported failure")
                rollouts = json.loads((training / "timing.json").read_text())["rollouts"]
                digest = result["result"]["final_policy_hash"]
                if expected_hash is None:
                    expected_hash = digest
                if digest != expected_hash:
                    raise RuntimeError("final policy hash differs")
                sample = sum(x["rollout_seconds"] for x in rollouts)
                update = sum(x["update_seconds"] for x in rollouts)
                report["runs"].append({"pair": pair, "mode": mode, "training": str(training),
                    "sample_seconds": sample, "update_seconds": update,
                    "loop_seconds": sample + update, "lifecycle_seconds": result["total_seconds"],
                    "final_policy_hash": digest})
                save()
                print(pair, mode, "sample", round(sample, 3), "loop", round(sample + update, 3), flush=True)
        report["summary"] = {}
        for metric in ("sample_seconds", "loop_seconds", "lifecycle_seconds"):
            groups = {mode: [r[metric] for r in report["runs"] if r["mode"] == mode]
                      for mode in ("baseline", "optimized")}
            a, b = (statistics.mean(groups[mode]) for mode in ("baseline", "optimized"))
            report["summary"][metric] = {"baseline_mean": a, "optimized_mean": b,
                "baseline_stdev": statistics.stdev(groups["baseline"]),
                "optimized_stdev": statistics.stdev(groups["optimized"]),
                "throughput_ratio": a / b, "time_reduction_fraction": 1 - b / a}
        report.update(status="PASS", final_hash_parity="PASS")
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()
        print("Report:", output, flush=True)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

"""Run the existing four-instance recipe with separate parent/worker profiles."""
import cProfile
import json
import os
from pathlib import Path
import pstats
import runpy
import sys
from datetime import datetime


def export_profile(source):
    stats = pstats.Stats(str(source))
    rows = []
    for (file, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        rows.append({"file": file, "line": line, "function": name,
                     "calls": calls, "primitive_calls": primitive,
                     "self_seconds": own, "inclusive_seconds": cumulative,
                     "callers": [{"file": k[0], "line": k[1], "function": k[2],
                                  "stats": list(v) if isinstance(v, tuple) else v}
                                 for k, v in callers.items()]})
    rows.sort(key=lambda x: x["inclusive_seconds"], reverse=True)
    source.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    root = Path(os.environ["SOKURL_RL_ROOT"])
    os.chdir(root)
    sys.path[:0] = [str(root / "src"), str(root / "tools")]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = root / "docs/reports/sampling_profile" / stamp
    output.mkdir(parents=True, exist_ok=False)
    worker = output / "worker.prof"
    training = root / "runs/training" / ("platform-profile-4-" + stamp)
    command = json.dumps([sys.executable.replace("\\", "/"), "-B", "-m", "cProfile",
                          "-o", worker.as_posix(), "tools/rollout_worker.py"])
    sys.argv = ["tools/train.py", "--config-name", "train_platform_initial",
                "runtime.command=" + command, "output=" + training.as_posix()]
    (output / "run.json").write_text(json.dumps({"training": str(training),
        "arguments": sys.argv, "scope": "parent main thread and worker main thread; wall time, not native CPU time",
        "overlap": "parent IPC waits overlap worker execution; never sum inclusive totals"}, indent=2), encoding="utf-8")
    profiler = cProfile.Profile()
    try:
        profiler.runcall(runpy.run_path, "tools/train.py", run_name="__main__")
    finally:
        profiler.dump_stats(str(output / "parent.prof"))
        export_profile(output / "parent.prof")
        if worker.exists():
            export_profile(worker)
        print("Profile directory:", output)


if __name__ == "__main__":
    main()

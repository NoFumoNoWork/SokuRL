"""Run a reproducible, paired strategy league using the Hydra config space."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from soku_rl.evaluation import make_plan, run_batch, summarize
from soku_rl.strategies import strategy_from_config
from game_batch import SokuGameBatch
import sokurl


ROOT = Path(__file__).resolve().parents[1]


def fingerprints():
    sources = sorted((ROOT / "src/soku_rl").glob("*.py")) + [
        ROOT / "tools" / name for name in (
            "game_batch.py", "evaluate.py", "bridge_shared.py", "sokurl.py",
            "headless_validation.py", "interaction_benchmark.py", "unlimited_benchmark.py")]
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    artifacts = [sokurl.GAME_DIR / name for name in
                 ("th123.exe", "th123a.dat", "th123b.dat", "th123c.dat", "d3d9.dll", "SWRSToys.ini")]
    artifacts += sorted((sokurl.GAME_DIR / "modules").rglob("*.dll"))
    artifacts += sorted((sokurl.GAME_DIR / "modules").rglob("*.ini"))
    artifacts += sorted((sokurl.GAME_DIR / "profile").glob("*.pf"))
    artifact_hashes = {str(p.relative_to(sokurl.GAME_DIR)).replace("\\", "/"):
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    implementation = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    game_id = hashlib.sha256(json.dumps(artifact_hashes, sort_keys=True).encode()).hexdigest()
    return {"implementation": implementation, "game_id": game_id,
            "source_hashes": source_hashes, "artifact_hashes": artifact_hashes}


def save_report(destination, report):
    temporary = destination.with_suffix(".pending.json")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(destination)


@hydra.main(version_base="1.3", config_path="../config", config_name="evaluation")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    if (config["workers"] < 1 or config["max_frames"] < 1
            or len(set(config["profiles"])) != len(config["profiles"])):
        raise ValueError("positive workers/frame limit and unique profiles required")
    sokurl._validate_game()
    identity = fingerprints()
    strategies = {name: strategy_from_config(name, config, identity["implementation"])
                  for name in config["profiles"]}
    protocol = {"game": identity["game_id"], "max_frames": config["max_frames"],
                "termination": "first knockout", "observation": "numeric_relative_v2",
                "reward": "terminal win +1 / loss -1 / double KO 0"}
    plan = make_plan(strategies, config["seeds"], config["policy_seed"], protocol)
    destination = ROOT / config["output"]
    if destination.exists():
        raise FileExistsError(f"choose a new output path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "success": False, "config": config, "identity": identity,
              "protocol": protocol, "strategies": {n: asdict(s) | {"fingerprint": s.fingerprint}
                                                    for n, s in strategies.items()},
              "plan": [asdict(t) for t in plan], "games": [], "batches": []}
    started = time.perf_counter()
    save_report(destination, report)
    try:
        for seed in config["seeds"]:
            trials = [t for t in plan if t.world_seed == seed]
            for start in range(0, len(trials), config["workers"]):
                batch_trials = trials[start:start + config["workers"]]
                print(f"seed={seed} batch={start} games={len(batch_trials)}", flush=True)
                batch_started = time.perf_counter()
                try:
                    batch = run_batch(SokuGameBatch(config["launch_timeout"]), batch_trials,
                                      strategies, config["max_frames"])
                except Exception as error:
                    report["games"].extend(asdict(t) | {"status": "error", "error": repr(error)}
                                           for t in batch_trials)
                    raise
                report["games"].extend(batch.pop("games"))
                batch["total_seconds"] = time.perf_counter() - batch_started
                report["batches"].append(batch)
                report["summary"] = summarize(plan, report["games"], config["alpha"])
                save_report(destination, report)
                print(f"completed={len(report['games'])}/{len(plan)} steps={batch['simulation_steps']} "
                      f"seconds={batch['total_seconds']:.2f}", flush=True)
        report["success"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        report["summary"] = summarize(plan, report["games"], config["alpha"])
        report["simulation_steps"] = sum(b["simulation_steps"] for b in report["batches"])
        save_report(destination, report)
        print(f"report={destination}", flush=True)


if __name__ == "__main__":
    main()

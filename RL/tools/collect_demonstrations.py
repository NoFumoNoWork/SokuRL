"""Collect independent rule demonstrations under the exact shared BR interface."""
from contextlib import closing
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="collect_demonstrations")
def main(cfg):
    from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
    from soku_rl.env.worker_pipe import WorkerBackend
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface, LearningVectorEnv
    from soku_rl.policy.loader import load_policy
    from soku_rl.policy.matchups import opponent_interface
    from soku_rl.rl import configure_runtime, ppo_settings, validate_payoff
    from soku_rl.rl.learner import learner_kind
    from soku_rl.rl.demonstrations import collect_demonstrations, demonstration_plan

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    ppo_settings(config)
    configure_runtime(config["rl"])
    if (config["algorithm"]["name"] != "br" or config["algorithm"]["matchups"]["mode"] != "sampled"
            or config["algorithm"][learner_kind(config["algorithm"])]["gamma"] != 1. or config["teacher"]["kind"] != "rule"
            or "curriculum" in config["algorithm"]):
        raise ValueError("rule demonstrations require sampled BR, gamma=1 and explicit fixed opponents")
    interface = LearningInterface(EpisodeConfig.from_dict(config["episode"]), LearningConfig(**config["wrappers"]))
    validate_payoff(interface, config["algorithm"])
    learner = config["algorithm"]["matchups"]["learner"]
    population = config["algorithm"]["opponents"]
    excluded = {seed for block in config["excluded"].values() for seed in block["world_seeds"]}
    prior_plans, prior_worlds = [], set()
    for name in config["excluded_datasets"]:
        path = Path(name).resolve(strict=True) / "episodes/plan.json"
        data = path.read_bytes()
        prior_worlds.update(row["world_seed"] for row in json.loads(data))
        prior_plans.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
    config["excluded"]["prior_collections"] = {"world_seeds": sorted(prior_worlds)}
    excluded.update(prior_worlds)
    plan = demonstration_plan(config["collection"], population, config["seed"], excluded)
    teacher = load_policy("teacher", config["teacher"], interface, config["device"])
    if config["behavior"] == {"kind": "teacher"}:
        behavior = teacher
    elif config["behavior"]["kind"] in {"sb3", "sb3_recurrent", "sb3_dqn", "teacher_takeover", "action_noise"}:
        import torch
        if config["behavior"]["kind"] == "action_noise" and config["behavior"]["policy"] != config["teacher"]:
            raise ValueError("noisy collection must wrap the declared teacher")
        if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for collection but unavailable")
        behavior = load_policy("behavior", config["behavior"], interface, config["device"])
    else:
        raise ValueError("collection behavior must be teacher, learned checkpoint, teacher_takeover or action_noise")
    opponents = [load_policy(entry["name"], entry["policy"],
        opponent_interface(interface, learner, entry), config["device"]) for entry in population]
    destination = Path(config["output"]).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    identity = {"source_hashes": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (root / "src/soku_rl", root / "tools") for p in sorted(folder.rglob("*.py"))},
        "sampling": ("teacher controls" if behavior is teacher else
            "same teacher labels before independently seeded uniform input replacement" if
            config["behavior"]["kind"] == "action_noise" else
            "learner prefix then same shadow teacher controls; supervise executed suffix" if
            config["behavior"]["kind"] == "teacher_takeover" else "learner controls; teacher only labels"),
        "behavior_fingerprint": behavior.fingerprint, "excluded_plans": prior_plans,
        "device": config["device"], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {name: version(name) for name in ["torch", "numpy", "gymnasium", "lupa",
            *(["stable-baselines3"] if behavior is not teacher else []),
            *(["sb3-contrib"] if config["behavior"]["kind"] in {"sb3_recurrent", "teacher_takeover"} else [])]}}
    (destination / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    report = {"success": False, "method": "rule_demonstrations"}
    started = time.perf_counter()
    try:
        with closing(WorkerBackend(log_path=destination / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(interface.episode.backend_observation())
            identity["runtime"] = backend.identity
            (destination / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
            env = LearningVectorEnv(TwoPlayerVectorEnv(backend, config["num_envs"], interface.episode), interface.config)
            report["result"] = collect_demonstrations(env, plan, learner, population, teacher,
                opponents, behavior, destination / "episodes")
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (destination / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

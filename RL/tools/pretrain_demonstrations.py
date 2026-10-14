"""Initialize the shared learner from complete, independently split rule demonstrations."""
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="pretrain_demonstrations")
def main(cfg):
    import torch
    from soku_rl.env import EpisodeConfig
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
    from soku_rl.rl import configure_runtime, ppo_settings
    from soku_rl.rl.behavior_cloning import fit_demonstrations
    from soku_rl.rl.demonstration_sets import load_demonstration_sets

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    configure_runtime(config["rl"])
    ppo_settings(config)
    if config["algorithm"]["name"] != "br" or "curriculum" in config["algorithm"]:
        raise ValueError("offline initialization requires BR without a sampling curriculum")
    interface = LearningInterface(EpisodeConfig.from_dict(config["episode"]), LearningConfig(**config["wrappers"]))
    sources = [config["pretraining"]["dataset"], *config["pretraining"]["additional_datasets"]]
    samples, training, datasets = load_demonstration_sets(sources, interface, config["pretraining"]["value_coef"])
    # Keep the dataset's character and opponent identities for paired BR evaluation.
    for key in ("matchups", "opponents"):
        config["algorithm"][key] = training["algorithm"][key]
    config["teacher"] = training["teacher"]
    config["training_method"] = "behavior_cloning"
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for pretraining but unavailable")
    destination = Path(config["output"]).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    identity = {"source_hashes": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (root / "src/soku_rl", root / "tools") for p in sorted(folder.rglob("*.py"))},
        "datasets": datasets, "teacher_fingerprint": datasets[0]["teacher_fingerprint"],
        "packages": {name: version(name) for name in ["torch", "numpy", "gymnasium", "stable-baselines3",
            *(["sb3-contrib"] if config["rl"]["policy_type"] == "lstm" else [])]},
        "device": config["device"], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(config["device"]) if str(config["device"]).startswith("cuda") else "cpu"}
    (destination / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    report = {"success": False, "algorithm": "br", "method": "behavior_cloning",
        "demonstration_env_steps": sum(dataset["environment_steps"] for dataset in datasets)}
    started = time.perf_counter()
    try:
        report["result"] = fit_demonstrations(interface, config["algorithm"], samples,
            config["pretraining"], config["device"], config["seed"], destination)
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (destination / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

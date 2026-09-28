"""Run BenchMARL IPPO with separate player groups and owned game factories."""
from dataclasses import asdict
import hashlib
import json

from benchmarl.algorithms import IppoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import CnnConfig, MlpConfig

from .benchmarl_task import SokuTask


def _apply(config, values):
    for key, value in values.items():
        if not hasattr(config, key):
            raise ValueError(f"unsupported {type(config).__name__} field: {key}")
        setattr(config, key, value)
    return config


def _policy_hash(policy):
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_benchmarl(config, device, directory):
    algorithm = config["algorithm"]
    experiment_config = _apply(ExperimentConfig.get_from_yaml(), algorithm["experiment"])
    experiment_config.sampling_device = str(device)
    experiment_config.train_device = str(device)
    experiment_config.buffer_device = "cpu"
    experiment_config.on_policy_n_envs_per_worker = config["num_envs"]
    experiment_config.save_folder = str(directory)
    algo_config = _apply(IppoConfig.get_from_yaml(), algorithm["ippo"])
    if config["episode"]["observation_mode"] == "image":
        model = _apply(CnnConfig.get_from_yaml(), algorithm["cnn"])
    else:
        model = _apply(MlpConfig.get_from_yaml(), algorithm["mlp"])
    task = SokuTask("VS", {"runtime": config["runtime"], "episode": config["episode"],
                          "log_directory": str(directory / "workers")})
    (directory / "benchmarl-config.json").write_text(json.dumps({
        "experiment": asdict(experiment_config), "algorithm": asdict(algo_config),
        "model": asdict(model)}, indent=2, default=str), encoding="utf-8")
    experiment = Experiment(task=task, algorithm_config=algo_config, model_config=model,
                            seed=config["seed"], config=experiment_config)
    try:
        initial = _policy_hash(experiment.policy)
    except BaseException:
        experiment.close()
        raise
    # BenchMARL 1.5.1 run() closes its collector, test env and logger on both
    # completion and failure. A second close can fail on already closed workers.
    experiment.run()
    final = _policy_hash(experiment.policy)
    if initial == final:
        raise RuntimeError("training ended without a change to policy parameters")
    return {"frames": experiment.total_frames, "initial_policy_hash": initial,
            "final_policy_hash": final, "experiment_directory": str(experiment.folder_name)}

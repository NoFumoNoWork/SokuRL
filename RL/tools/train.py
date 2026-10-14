"""Hydra entry point for public RL algorithms over the two-player game."""
import hashlib
from contextlib import closing
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def main(cfg: DictConfig):
    # The environment worker needs no CUDA library. Only this native learner does.
    import numpy as np
    import torch
    from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
    from soku_rl.env.worker_pipe import WorkerBackend
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface, LearningVectorEnv
    from soku_rl.rl import configure_runtime, ppo_settings, validate_payoff

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    learner = ppo_settings(config)
    configure_runtime(config["rl"])
    algorithm = config["algorithm"]["name"]
    if algorithm == "br":
        from soku_rl.marl.br import train_br as train
        dependencies = ["stable-baselines3"]
    elif algorithm == "nfsp":
        from soku_rl.marl.nfsp import train_nfsp as train
        dependencies = ["stable-baselines3"]
    elif algorithm == "psro":
        from soku_rl.marl.psro import train_psro as train
        dependencies = ["open-spiel", "stable-baselines3", "cvxpy"]
    elif algorithm == "ippo":
        from soku_rl.marl.ippo import train_ippo as train
        dependencies = ["stable-baselines3"]
    elif algorithm == "ppo":
        from soku_rl.rl.training import train_ppo as train
        dependencies = ["stable-baselines3"]
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    if learner["policy_type"] == "lstm":
        dependencies.append("sb3-contrib")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**31:
        raise ValueError("training seed must be in [0, 2**31)")
    if type(config["num_envs"]) is not int or config["num_envs"] < 1:
        raise ValueError("num_envs must be a positive integer")
    if not isinstance(config["runtime"]["command"], list):
        raise ValueError("runtime.command must be an explicit argument list")
    episode = EpisodeConfig.from_dict(config["episode"])
    learning = LearningConfig(**config["wrappers"])
    interface = LearningInterface(episode, learning)
    if algorithm == "nfsp" and (interface.observation_space.shape is None
                               or len(interface.observation_space.shape) != 1):
        raise ValueError("the NFSP average-policy reservoir requires numeric state observations")
    if algorithm == "ppo" and episode.observation_mode == "image":
        raise ValueError("fixed-rule PPO requires state observations; image self-play uses IPPO or PSRO")
    validate_payoff(interface, learner)
    if config["track"] == "human" and episode.observation_mode in {"diagnostic_state", "privileged_state"}:
        raise ValueError("the human track cannot expose privileged diagnostic state")
    if config["track"] not in {"human", "superhuman"}:
        raise ValueError("unsupported training track")
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
        torch.cuda.get_device_properties(device)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    destination = Path(config["output"]).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    platform = Path(os.environ.get("SOKURL_PLATFORM_ROOT", root.parent))
    sources = [*sorted((root / "src/soku_rl").rglob("*.py")), *sorted((root / "tools").rglob("*.py")), *sorted((platform / "tools").rglob("*.py"))]
    identity = {"source_hashes": {("RL/" + p.relative_to(root).as_posix() if p.is_relative_to(root)
                                 else "Platform/" + p.relative_to(platform).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sources},
                "packages": {p: version(p) for p in ["torch", "gymnasium", "pettingzoo", *dependencies]},
                "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "hydra_output": HydraConfig.get().runtime.output_dir}
    (destination / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    started = time.perf_counter()
    report = {"success": False, "algorithm": algorithm}
    try:
        with closing(WorkerBackend(log_path=destination / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(episode.backend_observation())
            identity["runtime"] = backend.identity
            (destination / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
            env = LearningVectorEnv(TwoPlayerVectorEnv(backend, config["num_envs"], episode), learning)
            report["result"] = train(env, config["algorithm"], device, config["seed"], destination)
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        (destination / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

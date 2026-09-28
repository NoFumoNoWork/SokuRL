"""Hydra entry point for public NFSP or PSRO over the two-player vector game."""
import hashlib
from contextlib import closing
from importlib.metadata import version
import json
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
    from soku_rl.worker_pipe import WorkerBackend

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    algorithm = config["algorithm"]["name"]
    if algorithm == "nfsp":
        from soku_rl.nfsp import train_nfsp as train
        dependencies = ["rlcard"]
    elif algorithm == "psro":
        from soku_rl.psro import train_psro as train
        dependencies = ["open-spiel", "stable-baselines3", "cvxpy"]
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**31:
        raise ValueError("training seed must be in [0, 2**31)")
    if type(config["num_envs"]) is not int or config["num_envs"] < 1:
        raise ValueError("num_envs must be a positive integer")
    if not isinstance(config["runtime"]["command"], list):
        raise ValueError("runtime.command must be an explicit argument list")
    episode = EpisodeConfig(**config["episode"])
    if episode.observation_mode == "diagnostic_state":
        raise ValueError("privileged diagnostic state is not a training observation")
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
    sources = [*sorted((root / "src/soku_rl").rglob("*.py")), *sorted((root / "tools").glob("*.py"))]
    identity = {"source_hashes": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sources},
                "packages": {p: version(p) for p in ["torch", "gymnasium", "pettingzoo", *dependencies]},
                "device": str(device), "hydra_output": HydraConfig.get().runtime.output_dir}
    (destination / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    started = time.perf_counter()
    report = {"success": False, "algorithm": algorithm}
    try:
        with closing(WorkerBackend(log_path=destination / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(episode.backend_observation())
            identity["runtime"] = backend.identity
            (destination / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
            env = TwoPlayerVectorEnv(backend, config["num_envs"], episode)
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

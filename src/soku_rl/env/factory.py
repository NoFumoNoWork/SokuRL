"""Construct an owned PettingZoo game with the configured transport and episode."""
from pathlib import Path
from uuid import uuid4

from soku_rl.worker_pipe import WorkerBackend
from .hisouten_env import EpisodeConfig, HisoutenParallelEnv


def make_pettingzoo_env(runtime, episode, log_directory):
    config = EpisodeConfig(**episode)
    if config.observation_mode == "diagnostic_state":
        raise ValueError("training factories do not expose privileged diagnostic state")
    log_path = Path(log_directory) / f"worker-{uuid4().hex}.log"
    backend = WorkerBackend(log_path=log_path, **runtime)
    try:
        backend.configure_observation(config.observation_mode)
        return HisoutenParallelEnv(backend, config)
    except BaseException:
        backend.close()
        raise

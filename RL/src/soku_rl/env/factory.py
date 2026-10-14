"""Construct an owned PettingZoo game with the configured transport and episode."""
from pathlib import Path
from uuid import uuid4
import json

from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.hisouten_env import EpisodeConfig, HisoutenParallelEnv


def make_pettingzoo_env(runtime, episode, log_directory):
    config = EpisodeConfig.from_dict(episode)
    log_path = Path(log_directory) / f"worker-{uuid4().hex}.log"
    backend = WorkerBackend(log_path=log_path, **runtime)
    try:
        backend.configure_observation(config.backend_observation())
        log_path.with_suffix(".json").write_text(json.dumps({"runtime": backend.identity,
            "episode": episode}, indent=2), encoding="utf-8")
        return HisoutenParallelEnv(backend, config)
    except BaseException:
        backend.close()
        raise

"""Evaluate final training policies using their saved observation and control contract."""
import hashlib
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from benchmark import run
from soku_rl.policy.artifacts import completed_policies


@hydra.main(version_base="1.3", config_path="../config", config_name="benchmark_training")
def main(cfg):
    directory = Path(cfg.training_directory).resolve(strict=True)
    training, candidate = completed_policies(directory)
    # Preserve the source episode and model interface. Runtime, GPU, seed split,
    # batch size and output location belong to this separate evaluation run.
    values = {key: training[key] for key in ("algorithm", "episode", "wrappers", "track")}
    values.update(candidate=candidate, training_directory=str(directory),
                  training_result_sha256=hashlib.sha256((directory / "result.json").read_bytes()).hexdigest())
    prepared = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key, value in values.items():
        prepared[key] = value
    run(prepared)


if __name__ == "__main__":
    main()

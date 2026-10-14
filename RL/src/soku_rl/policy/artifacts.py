"""Locate both final seat policies from one successfully completed training run."""
import json
from pathlib import Path

from omegaconf import OmegaConf


def completed_policies(directory):
    directory = Path(directory).resolve(strict=True)
    report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if report["success"] is not True:
        raise ValueError("final-policy evaluation requires successful completed training")
    config_path = directory / "config.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True, throw_on_missing=True)
    algorithm = config["algorithm"]
    if report["algorithm"] != algorithm["name"]:
        raise ValueError("training report and algorithm configuration differ")
    candidate = {"name": directory.name}
    for seat in ("player_0", "player_1"):
        spec = {"training_config": str(config_path)}
        shared = report["result"].get("format") == "sokurl-shared-ppo-v1"
        if algorithm["name"] == "ppo" or shared:
            from soku_rl.rl.learner import artifact_kind
            kind = "sb3" if algorithm["name"] == "nfsp" else artifact_kind(algorithm)
            spec.update(kind=kind, path=str(directory / seat / "final.zip"))
        elif algorithm["name"] == "nfsp":
            spec.update(kind="nfsp_average", player=seat, path=str(directory / "final" / f"{seat}.pt"))
        elif algorithm["name"] == "psro":
            spec.update(kind="psro_mixture", player=seat, path=str(directory / "population.json"))
        elif algorithm["name"] == "ippo":
            result = report["result"]
            experiment = directory / Path(result["experiment_directory"]).name
            spec.update(kind="benchmarl_ippo", player=seat,
                        path=str(experiment / "checkpoints" / f"checkpoint_{result['frames']}.pt"),
                        model_config=str(directory / "benchmarl-config.json"))
            if not Path(spec["model_config"]).is_file():
                raise FileNotFoundError(spec["model_config"])
        else:
            raise ValueError("unsupported completed training algorithm")
        if not Path(spec["path"]).is_file():
            raise FileNotFoundError(spec["path"])
        candidate[seat] = spec
    return config, candidate

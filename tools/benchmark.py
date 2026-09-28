"""Evaluate a saved two-seat policy against the fixed rule roster."""
from contextlib import closing
import hashlib
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="benchmark")
def main(cfg):
    import torch
    from soku_rl.checkpoint_policy import SeatPolicies, load_policy
    from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
    from soku_rl.observed_rules import RulePolicy
    from soku_rl.policy_benchmark import benchmark
    from soku_rl.worker_pipe import WorkerBackend

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    episode = EpisodeConfig(**config["episode"])
    if config["track"] == "human" and episode.observation_mode != "state":
        raise ValueError("this rule benchmark requires the human state observation")
    if config["track"] == "superhuman" and episode.observation_mode != "diagnostic_state":
        raise ValueError("the superhuman benchmark requires diagnostic state")
    opponents = config["benchmark"]["opponents"]
    if len(opponents) != len(set(opponents)) or "idle" in opponents:
        raise ValueError("the benchmark roster must contain distinct non-idle rules")
    candidate = config["candidate"]
    if candidate["name"] in opponents:
        raise ValueError("candidate name collides with an opponent")
    device = torch.device(config["device"])
    roles = tuple(load_policy(candidate["name"], candidate[a], episode, device)
                  for a in ("player_0", "player_1"))
    strategies = {candidate["name"]: SeatPolicies(candidate["name"], roles)}
    source = Path(__file__).resolve().parents[1] / "src/soku_rl"
    implementation = hashlib.sha256(b"".join((source / name).read_bytes() for name in (
        "observed_rules.py", "baselines.py", "community_rules.py", "strategies.py"))).hexdigest()
    for name in opponents:
        rule = RulePolicy(name, config["rules"], episode, implementation)
        strategies[name] = SeatPolicies(name, (rule, rule))
    directory = Path(config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    report = {"success": False}
    try:
        with closing(WorkerBackend(log_path=directory / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(episode.backend_observation())
            report["runtime"] = backend.identity
            env = TwoPlayerVectorEnv(backend, config["num_envs"], episode)
            report["result"] = benchmark(env, strategies, candidate["name"], config["benchmark"],
                backend.identity["fingerprints"]["game_id"], directory)
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

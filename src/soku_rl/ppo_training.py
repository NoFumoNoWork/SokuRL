"""Train one SB3 PPO policy per seat against a fixed rule population."""
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from .observed_rules import RulePolicy
from .strategies import rule_implementation
from .learning_wrappers import LearningRulePolicy
from .ppo_response import OpponentMixtureVecEnv
from .checkpoint_policy import read_training_contract


def parameter_hash(policy):
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_ppo(algorithm, policy_type, env, interface, config, source, device, seed):
    if source == {"kind": "fresh"}:
        return algorithm(policy_type, env, seed=seed, device=device, **config["ppo"]), source
    if set(source) != {"kind", "path", "training_config"} or source["kind"] != "checkpoint":
        raise ValueError("initial policy must be fresh or an explicit training checkpoint")
    previous = read_training_contract(source["training_config"], interface)["algorithm"]
    if any(previous[key] != config[key] for key in ("name", "policy_type", "ppo", "timeout_payoff")):
        raise ValueError("continued PPO must retain its algorithm and optimizer configuration")
    path = Path(source["path"]).resolve(strict=True)
    model = algorithm.load(path, env=env, device=device)
    model.set_random_seed(seed)
    return model, source | {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class EpisodeRecords(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.records = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.records.append({key: info[key] for key in (
                    "episode", "frame", "outcome", "decision_frames", "latency_frames", "training_context")})
        return True

    def _on_rollout_end(self):
        (self.directory / "progress.json").write_text(json.dumps({
            "steps": self.num_timesteps, "episodes": self.records}, indent=2), encoding="utf-8")


def train_ppo(env, config, device, seed, directory):
    if config["policy_type"] == "lstm":
        from sb3_contrib import RecurrentPPO
        algorithm, policy_type = RecurrentPPO, "MlpLstmPolicy"
    elif config["policy_type"] == "mlp":
        algorithm, policy_type = PPO, "MlpPolicy"
    else:
        raise ValueError("PPO policy_type must be mlp or lstm")
    if config["timeout_payoff"] != "zero_at_horizon":
        raise ValueError("PPO requires the declared finite-horizon payoff")
    if config["players"] != [0, 1]:
        raise ValueError("train both seat policies for seat-swapped evaluation")
    if config["timesteps_per_player"] < 1 or config["checkpoint_every"] < env.num_envs:
        raise ValueError("invalid PPO training or checkpoint interval")
    if not config["opponents"] or len(set(config["opponents"])) != len(config["opponents"]):
        raise ValueError("fixed opponent roster must be nonempty and distinct")
    episode = env.episodes[0].config
    source = rule_implementation()
    opponents = [LearningRulePolicy(RulePolicy(name, config["rules"], episode, source), env.interface)
                 for name in config["opponents"]]
    weights = np.full(len(opponents), 1 / len(opponents))
    results = {}
    for player in config["players"]:
        destination = directory / f"player_{player}"
        destination.mkdir()
        view = OpponentMixtureVecEnv(env, player, opponents, weights, seed + player)
        try:
            model, source = initialize_ppo(algorithm, policy_type, view, env.interface, config,
                config["initial_policies"][f"player_{player}"], device, seed + player)
            model.set_logger(configure(str(destination / "scalars"), ["csv", "stdout"]))
            initial = parameter_hash(model.policy)
            start_steps = model.num_timesteps
            callbacks = CallbackList([
                EpisodeRecords(destination),
                CheckpointCallback(save_freq=config["checkpoint_every"] // env.num_envs,
                                   save_path=str(destination / "checkpoints"), name_prefix="ppo"),
            ])
            model.learn(total_timesteps=config["timesteps_per_player"], callback=callbacks,
                        reset_num_timesteps=False)
            final = parameter_hash(model.policy)
            if initial == final:
                raise RuntimeError("PPO completed without a policy update")
            path = destination / "final.zip"
            model.save(path)
            results[f"player_{player}"] = {"steps": model.num_timesteps, "start_steps": start_steps,
                "additional_steps": model.num_timesteps - start_steps, "initial_policy": source,
                "initial_policy_hash": initial, "final_policy_hash": final,
                "checkpoint": str(path), "opponents": {p.name: p.fingerprint for p in opponents}}
        finally:
            view.close()
    return results

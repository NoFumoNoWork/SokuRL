"""Load trusted training artifacts as per-episode categorical policies."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from soku_rl.policy.contract import read_training_contract
from soku_rl.policy.base import RLPolicy
from soku_rl.policy.population import PPOPolicy, UniformPolicy, MixturePolicy
from soku_rl.policy.loader import load_policy


class NetworkPolicy(RLPolicy):
    def __init__(self, name, network, shape, num_actions, identity, device):
        self.name, self.shape, self.device = name, tuple(shape), device
        self.network = network.to(device).eval()
        self.fingerprint = identity
        self.num_actions = num_actions

    def spawn(self, seed):
        return NetworkEpisode(self, np.random.default_rng(seed))


@dataclass
class NetworkEpisode:
    policy: NetworkPolicy
    rng: object

    @torch.inference_mode()
    def act(self, observation):
        if observation.shape != self.policy.shape or not np.isfinite(observation).all():
            raise ValueError("observation does not match the loaded policy")
        value = torch.as_tensor(observation, dtype=torch.float32, device=self.policy.device).unsqueeze(0)
        probabilities = self.policy.network(value).softmax(-1)[0].cpu().numpy().astype(np.float64)
        if probabilities.shape != (self.policy.num_actions,) or not np.isfinite(probabilities).all():
            raise RuntimeError("policy produced invalid probabilities")
        probabilities /= probabilities.sum()
        return int(self.rng.choice(self.policy.num_actions, p=probabilities))


def _mlp(widths, activation):
    layers = []
    for index, (input_size, output_size) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        layers.append(nn.Linear(input_size, output_size))
        if index < len(widths) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def load_population(name, spec, interface, device, path, identity):
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["format"] != "sokurl-psro-population-v1" or spec["player"] not in {"player_0", "player_1"}:
        raise ValueError("PSRO population format or seat differs")
    if len(saved["populations"]) != 2 or len(saved["meta_strategies"]) != 2:
        raise ValueError("PSRO population must contain both seats")
    player = ("player_0", "player_1").index(spec["player"])
    members = []
    for entry in saved["populations"][player]:
        if entry["kind"] == "uniform":
            if entry["num_actions"] != interface.action_space.n:
                raise ValueError("PSRO member action space differs")
            member = UniformPolicy(entry["name"], entry["num_actions"])
        elif entry["kind"] in {"sb3", "sb3_recurrent", "sb3_dqn"}:
            member = load_policy(entry["name"], {
                "kind": entry["kind"], "path": str(path.parent / entry["path"]),
                "training_config": spec["training_config"]}, interface, device)
        else:
            raise ValueError("unsupported PSRO population member")
        if member.fingerprint != entry["fingerprint"]:
            raise ValueError("PSRO member fingerprint differs from the saved population")
        members.append(member)
    identity = hashlib.sha256((identity + spec["player"]).encode()).hexdigest()
    return MixturePolicy(name, members, saved["meta_strategies"][player], identity)


def load_checkpoint(name, spec, interface, device):
    read_training_contract(spec["training_config"], interface)
    path = Path(spec["path"]).resolve(strict=True)
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    shape, num_actions = interface.observation_space.shape, interface.action_space.n
    if spec["kind"] == "psro_mixture":
        return load_population(name, spec, interface, device, path, identity)
    if spec["kind"] in {"sb3", "sb3_recurrent", "sb3_dqn"}:
        if spec["kind"] == "sb3_dqn":
            from soku_rl.rl.dqn import DoubleDQN as Algorithm
            from soku_rl.policy.dqn import DQNPolicy as Policy
        elif spec["kind"] == "sb3_recurrent":
            from sb3_contrib import RecurrentPPO as Algorithm
            from soku_rl.policy.recurrent import RecurrentPPOPolicy as Policy
        else:
            from stable_baselines3 import PPO as Algorithm
            Policy = PPOPolicy
        model = Algorithm.load(path, device=device)
        if model.observation_space != interface.observation_space or model.action_space.n != num_actions:
            raise ValueError("SB3 checkpoint and evaluation spaces differ")
        return Policy(name, model, path)
    # These files are artifacts from our own training, not untrusted uploads.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if shape is None or len(shape) != 1:
        raise ValueError("this checkpoint loader supports numeric observations")
    with torch.random.fork_rng(devices=[]):
        if spec["kind"] == "nfsp_average":
            if saved["format"] != "sokurl-openspiel-nfsp-v1" or tuple(saved["observation_shape"]) != shape:
                raise ValueError("NFSP checkpoint format or observation shape differs")
            if saved["num_actions"] != num_actions or saved["player"] != spec["player"]:
                raise ValueError("NFSP checkpoint action space or seat differs")
            widths = [shape[0], *saved["agent_config"]["hidden_layers_sizes"], num_actions]
            network = _mlp(widths, nn.ReLU)
            weights = {}
            for i in range(len(widths) - 1):
                prefix = f"model.{i}.0" if i < len(widths) - 2 else f"model.{i}"
                for field in ("weight", "bias"):
                    weights[f"{2 * i}.{field}"] = saved["average_network"][f"{prefix}.{field}"]
        elif spec["kind"] == "benchmarl_ippo":
            config_path = Path(spec["model_config"]).resolve(strict=True)
            model = json.loads(config_path.read_text(encoding="utf-8"))["model"]
            if (model["activation_class"] != "<class 'torch.nn.modules.activation.Tanh'>"
                    or model["norm_class"] is not None or model["num_feature_dims"] != 1
                    or model["layer_class"] != "<class 'torch.nn.modules.linear.Linear'>"):
                raise ValueError("unsupported BenchMARL network architecture")
            identity = hashlib.sha256((identity + config_path.read_text(encoding="utf-8") + spec["player"]).encode()).hexdigest()
            network = _mlp([shape[0], *model["num_cells"], num_actions], nn.Tanh)
            prefix = "actor_network_params.module.0.mlp.params."
            weights = {key.removeprefix(prefix): value for key, value in saved[f"loss_{spec['player']}"].items()
                       if key.startswith(prefix)}
        else:
            raise ValueError("unsupported policy checkpoint kind")
        network.load_state_dict(weights, strict=True)
    return NetworkPolicy(name, network, shape, num_actions, identity, device)

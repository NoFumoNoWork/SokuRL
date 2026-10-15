"""Load a play opponent without importing training libraries for rules or ONNX."""
from dataclasses import replace
import json
from pathlib import Path

from gymnasium import spaces
import numpy as np
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.policy.base import RLPolicy, RulePolicy
from soku_rl.policy.loader import load_policy
from soku_rl.policy.population import MixturePolicy


def checkpoint_training(spec):
    if spec["kind"] in {"onnx", "onnx_recurrent", "onnx_dqn"}:
        path = Path(spec["path"]).resolve(strict=True)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        training_path = path.parent / manifest["training_config"]
    else:
        training_path = Path(spec["training_config"])
    return OmegaConf.to_container(OmegaConf.load(training_path), resolve=True)


def play_interface(candidate, episode, wrappers, track, overrides):
    """Use a saved model's complete input contract, with explicit live selections."""
    spec = candidate["policy"]
    if spec["kind"] != "rule":
        forbidden = [value for value in overrides if
            value.lstrip("+").split("=", 1)[0].startswith(("episode.", "wrappers"))]
        if forbidden:
            raise ValueError("saved opponents supply episode and wrappers; remove overrides: " + ", ".join(forbidden))
        training = checkpoint_training(spec)
        episode = training["episode"]
        # Schema-1 artifacts used the identity wrapper, as in the shared loader.
        wrappers = training["wrappers"] if "wrappers" in training else {
            "action_set": "full", "relative_features": False, "action_history": 0, "health_potential_scale": 0.}
    interface = LearningInterface(EpisodeConfig.from_dict(episode), LearningConfig(**wrappers))
    mode = interface.episode.observation_mode
    if mode not in {"state", "privileged_state", "diagnostic_state"}:
        raise ValueError("realtime play currently supports state, privileged_state and diagnostic_state observations")
    if track != ("superhuman" if mode in {"privileged_state", "diagnostic_state"} else "human"):
        raise ValueError("opponent observation contract differs from the selected track")
    return interface


def warm_play_policy(policy, interface, seed):
    """Warm every learned member using disposable memory before opening a game."""
    if isinstance(policy, MixturePolicy):
        return sum(warm_play_policy(member, interface, seed) for member in policy.members)
    if isinstance(policy, RulePolicy):
        return 0
    if not isinstance(policy, RLPolicy):
        raise TypeError("play preparation requires an RLPolicy, RulePolicy or MixturePolicy")

    def zeros(space):
        if isinstance(space, spaces.Dict):
            return {key: zeros(value) for key, value in space.spaces.items()}
        if not isinstance(space, spaces.Box):
            raise TypeError("unsupported policy observation space")
        return np.zeros(space.shape, dtype=space.dtype)

    observation, actor = zeros(interface.observation_space), policy.spawn(seed)
    for _ in range(8):
        interface.command(actor.act(observation))
    return 1


def checkpoint_interface(spec, interface):
    training = checkpoint_training(spec)
    match = EpisodeConfig.from_dict(training["episode"]).match
    # Character selection is a live match setting, not a tensor/clock change.
    # The shared checkpoint loader still checks every observation, timing and
    # wrapper field against the artifact's original training contract.
    return LearningInterface(replace(interface.episode, match=match), interface.config)


def load_play_policy(candidate, interface, rules, device, seat):
    spec = candidate["policy"]
    if seat not in (0, 1):
        raise ValueError("policy seat must be 0 or 1")
    if spec["kind"] == "rule":
        spec = spec | {"rules": rules}
    if spec["kind"] in {"sb3", "sb3_dqn", "sb3_recurrent", "nfsp_average", "psro_mixture", "benchmarl_ippo", "onnx", "onnx_recurrent", "onnx_dqn"}:
        interface = checkpoint_interface(spec, interface)
    if spec["kind"] in {"sb3", "sb3_dqn", "sb3_recurrent", "nfsp_average", "psro_mixture", "benchmarl_ippo"}:
        import torch
        torch.set_num_threads(1)
    return load_policy(candidate["name"], spec, interface, device)

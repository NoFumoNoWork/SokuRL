"""Check checkpoint logits, action spaces and per-episode recurrent memory."""
from dataclasses import asdict
import json

import numpy as np
from omegaconf import OmegaConf
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from test_env_timing import VISIBILITY
from soku_rl.policy.loader import load_policy
from soku_rl.env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface


def interface():
    episode = EpisodeConfig(7200, 1, 3, 12, "state", VISIBILITY, LEGACY_MATCH)
    return LearningInterface(episode, LearningConfig("combat", True, 8, 1.))


def training_config(tmp_path, contract):
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"episode": asdict(contract.episode), "wrappers": asdict(contract.config)}), path)
    return str(path)


def test_benchmarl_checkpoint_reconstructs_exact_actor_logits(tmp_path):
    contract = interface()
    model = nn.Sequential(nn.Linear(contract.observation_space.shape[0], 32), nn.Tanh(), nn.Linear(32, 90))
    path = tmp_path / "ippo.pt"
    torch.save({"loss_player_0": {"actor_network_params.module.0.mlp.params." + k: v
                                  for k, v in model.state_dict().items()}}, path)
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"model": {"num_cells": [32],
        "activation_class": "<class 'torch.nn.modules.activation.Tanh'>", "norm_class": None,
        "num_feature_dims": 1, "layer_class": "<class 'torch.nn.modules.linear.Linear'>"}}))
    loaded = load_policy("ippo", {"kind": "benchmarl_ippo", "path": str(path),
        "model_config": str(model_path), "training_config": training_config(tmp_path, contract),
        "player": "player_0"}, contract, "cpu")
    observation = torch.rand(3, contract.observation_space.shape[0])
    torch.testing.assert_close(model(observation), loaded.network(observation), rtol=0, atol=0)


def test_nfsp_checkpoint_reconstructs_upstream_network(tmp_path):
    pytest.importorskip("open_spiel")
    from open_spiel.python.pytorch.dqn import MLP
    contract = interface()
    model = MLP(contract.observation_space.shape[0], [32, 16], 90, seed=12)
    path = tmp_path / "nfsp.pt"
    torch.save({"format": "sokurl-openspiel-nfsp-v1", "observation_shape": contract.observation_space.shape,
        "num_actions": 90, "player": "player_0", "agent_config": {"hidden_layers_sizes": [32, 16]},
        "average_network": model.state_dict()}, path)
    loaded = load_policy("nfsp", {"kind": "nfsp_average", "path": str(path), "player": "player_0",
        "training_config": training_config(tmp_path, contract)}, contract, "cpu")
    observation = torch.rand(3, contract.observation_space.shape[0])
    torch.testing.assert_close(model(observation), loaded.network(observation), rtol=0, atol=0)


def test_saved_recurrent_policy_keeps_episode_memory_separate(tmp_path):
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import RecurrentPPO
    import gymnasium as gym

    contract = interface()
    env = gym.Env()
    env.observation_space, env.action_space = contract.observation_space, contract.action_space
    model = RecurrentPPO("MlpLstmPolicy", env, device="cpu", n_steps=2, batch_size=2,
                         policy_kwargs={"lstm_hidden_size": 16, "net_arch": [16]})
    path = tmp_path / "recurrent.zip"
    model.save(path)
    loaded = load_policy("recurrent", {"kind": "sb3_recurrent", "path": str(path),
        "training_config": training_config(tmp_path, contract)}, contract, "cpu")
    first, second = loaded.spawn(4), loaded.spawn(4)
    observation = np.ones(contract.observation_space.shape, np.float32) * .2
    action = first.act(observation)
    assert 0 <= action < 90
    assert all(torch.count_nonzero(state) == 0 for state in second.states)
    assert any(torch.count_nonzero(state) > 0 for state in first.states)
    assert second.act(observation) == action

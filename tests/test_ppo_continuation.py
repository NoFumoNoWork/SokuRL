"""Check that PPO continuation retains learned parameters and its input contract."""
from dataclasses import asdict, replace

import gymnasium as gym
from omegaconf import OmegaConf
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from soku_rl.learning_wrappers import LearningInterface
from soku_rl.ppo_training import initialize_ppo, parameter_hash
from test_policy_artifacts import interface


@pytest.mark.parametrize("policy_type", ["mlp", "lstm"])
def test_continuation_restores_optimizer_and_supports_new_vector_size(tmp_path, policy_type):
    if policy_type == "lstm":
        from sb3_contrib import RecurrentPPO
        algorithm, policy = RecurrentPPO, "MlpLstmPolicy"
        architecture = {"net_arch": [16], "lstm_hidden_size": 16}
    else:
        algorithm, policy, architecture = PPO, "MlpPolicy", {"net_arch": [16]}
    contract = interface()

    def make_env():
        env = gym.Env()
        env.observation_space, env.action_space = contract.observation_space, contract.action_space
        return env

    config = {"name": "ppo", "policy_type": policy_type, "timeout_payoff": "zero_at_horizon",
              "ppo": {"n_steps": 2, "batch_size": 2, "gamma": 1., "policy_kwargs": architecture}}
    initial, source = initialize_ppo(algorithm, policy, make_env(), contract, config,
                                     {"kind": "fresh"}, "cpu", 7)
    assert source == {"kind": "fresh"}
    # Create nonempty optimizer state without running a game or a short experiment.
    sum(parameter.sum() for parameter in initial.policy.parameters()).backward()
    initial.policy.optimizer.step()
    initial.num_timesteps = 64
    checkpoint = tmp_path / "policy.zip"
    initial.save(checkpoint)
    training = tmp_path / "training.yaml"
    OmegaConf.save(OmegaConf.create({"episode": asdict(contract.episode),
        "wrappers": asdict(contract.config), "algorithm": config}), training)
    spec = {"kind": "checkpoint", "path": str(checkpoint), "training_config": str(training)}
    env = DummyVecEnv([make_env, make_env])
    try:
        loaded, metadata = initialize_ppo(algorithm, policy, env, contract, config, spec, "cpu", 19)
        assert loaded.num_timesteps == 64 and loaded.n_envs == 2
        assert loaded._last_obs is None
        assert parameter_hash(loaded.policy) == parameter_hash(initial.policy)
        assert len(metadata["sha256"]) == 64
        expected = initial.policy.optimizer.state_dict()
        actual = loaded.policy.optimizer.state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        for parameter, values in expected["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(actual["state"][parameter][key], value, rtol=0, atol=0)
        changed = LearningInterface(replace(contract.episode, latency_frames=0), contract.config)
        with pytest.raises(ValueError, match="episode configurations differ"):
            initialize_ppo(algorithm, policy, env, changed, config, spec, "cpu", 19)
        with pytest.raises(ValueError, match="optimizer configuration"):
            initialize_ppo(algorithm, policy, env, contract,
                config | {"ppo": config["ppo"] | {"gamma": .9}}, spec, "cpu", 19)
    finally:
        env.close()

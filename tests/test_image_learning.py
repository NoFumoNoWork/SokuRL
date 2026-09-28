"""Check image/command observations across policy and environment adapters."""
from dataclasses import replace

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from test_env_timing import RecordingBackend, VISIBILITY
from soku_rl.env import EpisodeConfig, HisoutenParallelEnv, TwoPlayerVectorEnv
from soku_rl.env.encoding import AGENTS, decode_action
from soku_rl.learning_wrappers import LearningConfig, LearningParallelEnv, LearningVectorEnv
from soku_rl.pixels import RGBFrame


class ImageBackend(RecordingBackend):
    def _state(self, slot):
        state = super()._state(slot)
        image = RGBFrame(state.frame, 320, 240, bytes([state.frame % 256]) * (320 * 240 * 3))
        return replace(state, observations=(image, image))


def episode():
    return EpisodeConfig(6, 1, 3, 12, "image", VISIBILITY)


def learning():
    return LearningConfig("combat", False, 8, 0.)


def test_image_history_is_private_and_partial_reset_isolated():
    env = LearningVectorEnv(TwoPlayerVectorEnv(ImageBackend(), 2, episode()), learning())
    try:
        observations, _ = env.reset({0: 1, 1: 2})
        assert env.single_observation_space.contains(observations[0][AGENTS[0]])
        assert not observations[0][AGENTS[0]]["commands"].any()
        actions = {0: dict(zip(AGENTS, (89, 40))), 1: dict.fromkeys(AGENTS, 1)}
        observations, rewards, *_ = env.step(actions)
        for slot, players in observations.items():
            for agent, observation in players.items():
                assert env.single_observation_space.contains(observation)
                command = decode_action(env.interface.command(actions[slot][agent])).inputs
                np.testing.assert_array_equal(observation["commands"][-8:], command)
                assert (observation["image"][:3] == 3).all()
                assert rewards[slot][agent] == 0.
                np.testing.assert_array_equal(env.interface.base_observation(observation), observation["image"])
        before = list(env.transforms[1].history[AGENTS[0]])
        reset, _ = env.reset({0: 3})
        assert not reset[0][AGENTS[0]]["commands"].any()
        assert list(env.transforms[1].history[AGENTS[0]]) == before
    finally:
        env.close()


def test_image_dictionary_obeys_pettingzoo_contract():
    env = LearningParallelEnv(HisoutenParallelEnv(ImageBackend(), episode()), learning())
    try:
        parallel_api_test(env, num_cycles=1000)
    finally:
        env.close()


def test_torchrl_preserves_image_and_command_fields():
    pytest.importorskip("torchrl")
    from torchrl.envs.utils import check_env_specs
    from soku_rl.torchrl_env import wrap_torchrl
    env = wrap_torchrl(LearningParallelEnv(HisoutenParallelEnv(ImageBackend(), episode()), learning()), 3, "cpu")
    try:
        check_env_specs(env)
        value = env.reset()["player_0", "observation"]
        assert value["image"].shape == (1, 240, 320, 4)
        assert value["commands"].shape == (1, 64)
        assert not value["commands"].any()
    finally:
        env.close()


def test_benchmarl_cnn_accepts_both_image_and_command_inputs():
    pytest.importorskip("benchmarl")
    from benchmarl.models import CnnConfig
    from torchrl.data import Composite, Unbounded
    from soku_rl.benchmarl_task import SokuTask
    from soku_rl.torchrl_env import wrap_torchrl
    env = wrap_torchrl(LearningParallelEnv(HisoutenParallelEnv(ImageBackend(), episode()), learning()), 3, "cpu")
    try:
        group = "player_0"
        inputs = Composite({group: SokuTask("VS", {}).observation_spec(env)[group]})
        outputs = Composite({group: Composite({"logits": Unbounded(shape=(1, 90))}, shape=(1,))})
        config = CnnConfig.get_from_yaml()
        config.cnn_num_cells, config.cnn_kernel_sizes = [8, 8], [8, 4]
        config.cnn_strides, config.cnn_paddings, config.mlp_num_cells = [4, 2], [0, 0], [16]
        model = config.get_model(input_spec=inputs, output_spec=outputs, agent_group=group,
            input_has_agent_dim=True, n_agents=1, centralised=False, share_params=True,
            device="cpu", action_spec=env.action_spec)
        assert model(env.reset())[group, "logits"].shape == (1, 90)
        assert model.input_features_tensors == 64
    finally:
        env.close()


def test_sb3_image_view_keeps_terminal_observation_before_reset(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from soku_rl.population import UniformPolicy
    from soku_rl.ppo_response import OpponentMixtureVecEnv
    from soku_rl.checkpoint_policy import load_policy
    from test_policy_artifacts import training_config
    env = LearningVectorEnv(TwoPlayerVectorEnv(ImageBackend(), 1, episode()), learning())
    view = OpponentMixtureVecEnv(env, 0, [UniformPolicy("random", 90)], [1.], 4)
    try:
        # Exercise SB3's public mixed-image input path without a training run.
        model = PPO("MultiInputPolicy", view, device="cpu", n_steps=2, batch_size=2)
        obs = view.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1,)
        path = tmp_path / "image-policy.zip"
        model.save(path)
        saved = load_policy("image-policy", {"kind": "sb3", "path": str(path),
            "training_config": training_config(tmp_path, env.interface)}, env.interface, "cpu")
        assert 0 <= saved.spawn(5).act({k: v[0] for k, v in obs.items()}) < 90
        for _ in range(2):
            obs, rewards, dones, infos = view.step(np.array([89]))
        assert dones.tolist() == [True]
        assert rewards.tolist() == [0.]
        terminal = infos[0]["terminal_observation"]
        assert (terminal["image"][:3] == 6).all()
        assert (obs["image"][0, :3] == 0).all()
        assert terminal["commands"].any() and not obs["commands"].any()
        assert not infos[0]["TimeLimit.truncated"]
        assert infos[0]["source_truncated"]
    finally:
        view.close()
        env.close()

"""Adapt the public two-player PettingZoo game to TorchRL tensor dictionaries."""
from pettingzoo.utils.wrappers import BaseParallelWrapper
from torchrl.envs import PettingZooWrapper
from gymnasium import spaces
import numpy as np

from soku_rl.env.encoding import AGENTS
from soku_rl.env.factory import make_pettingzoo_env
from soku_rl.env.wrappers.learning import LearningConfig, LearningParallelEnv


class TorchRLInputs(BaseParallelWrapper):
    """Adapt numeric info, image layout and finite-horizon payoff for training.

    A public timeout is a terminal zero-payoff outcome in this training game.
    Mark it terminal here so GAE does not bootstrap a value beyond the horizon.
    The source timeout remains available in numeric info, never in policy input.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_spaces = {}
        for agent in env.possible_agents:
            self.observation_spaces[agent] = self.convert_space(env.observation_space(agent))

    @staticmethod
    def convert_space(space):
        if isinstance(space, spaces.Dict):
            return spaces.Dict({key: TorchRLInputs.convert_space(value) for key, value in space.items()})
        if space.dtype == np.uint8 and len(space.shape) == 3:
            channels, height, width = space.shape
            return spaces.Box(0, 1, (height, width, channels), np.float32)
        return space

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @staticmethod
    def observations(values):
        if isinstance(values, dict):
            return {key: TorchRLInputs.observations(value) for key, value in values.items()}
        return (np.moveaxis(values, 0, -1).astype(np.float32) / 255
                if values.dtype == np.uint8 and values.ndim == 3 else values)
    @staticmethod
    def convert(infos):
        keys = ("frame", "episode", "decision_frames", "latency_frames")
        return {agent: {key: info[key] for key in keys} | {"source_truncated": 0}
                for agent, info in infos.items()}

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self.observations(observations), self.convert(infos)

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        numeric = self.convert(infos)
        for agent in infos:
            numeric[agent]["source_truncated"] = int(truncated[agent])
        terminal = {agent: terminated[agent] or truncated[agent] for agent in terminated}
        return self.observations(observations), rewards, terminal, dict.fromkeys(truncated, False), numeric


def wrap_torchrl(env, seed, device):
    return PettingZooWrapper(TorchRLInputs(env), categorical_actions=True,
        group_map={agent: [agent] for agent in AGENTS}, use_mask=True,
        seed=seed, device=device)


def make_torchrl_env(runtime, episode, wrappers, log_directory, seed, device):
    env = make_pettingzoo_env(runtime, episode, log_directory)
    try:
        return wrap_torchrl(LearningParallelEnv(env, LearningConfig(**wrappers)), seed, device)
    except BaseException:
        env.close()
        raise

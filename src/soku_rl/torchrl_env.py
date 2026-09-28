"""Adapt the public two-player PettingZoo game to TorchRL tensor dictionaries."""
from pettingzoo.utils.wrappers import BaseParallelWrapper
from torchrl.envs import PettingZooWrapper
from gymnasium import spaces
import numpy as np

from .env.encoding import AGENTS
from .env.factory import make_pettingzoo_env
from .learning_wrappers import LearningConfig, LearningParallelEnv


class TorchRLInputs(BaseParallelWrapper):
    """Convert info to numbers and CHW image bytes to BenchMARL's HWC floats."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_spaces = {}
        for agent in env.possible_agents:
            space = env.observation_space(agent)
            if space.dtype == np.uint8 and len(space.shape) == 3:
                channels, height, width = space.shape
                space = spaces.Box(0, 1, (height, width, channels), np.float32)
            self.observation_spaces[agent] = space

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @staticmethod
    def observations(values):
        return {agent: np.moveaxis(value, 0, -1).astype(np.float32) / 255
                if value.dtype == np.uint8 and value.ndim == 3 else value
                for agent, value in values.items()}
    @staticmethod
    def convert(infos):
        keys = ("frame", "episode", "decision_frames", "latency_frames")
        return {agent: {key: info[key] for key in keys} for agent, info in infos.items()}

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self.observations(observations), self.convert(infos)

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        return self.observations(observations), rewards, terminated, truncated, self.convert(infos)


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

"""Adapt the public two-player PettingZoo game to TorchRL tensor dictionaries."""
from pettingzoo.utils.wrappers import BaseParallelWrapper
from torchrl.envs import PettingZooWrapper

from .env.encoding import AGENTS
from .env.factory import make_pettingzoo_env


class NumericInfo(BaseParallelWrapper):
    """TorchRL 0.10 requires numeric info; outcome labels stay in game records."""
    @staticmethod
    def convert(infos):
        keys = ("frame", "episode", "decision_frames", "latency_frames")
        return {agent: {key: info[key] for key in keys} for agent, info in infos.items()}

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return observations, self.convert(infos)

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        return observations, rewards, terminated, truncated, self.convert(infos)


def wrap_torchrl(env, seed, device):
    return PettingZooWrapper(NumericInfo(env), categorical_actions=True,
        group_map={agent: [agent] for agent in AGENTS}, use_mask=True,
        seed=seed, device=device)


def make_torchrl_env(runtime, episode, log_directory, seed, device):
    env = make_pettingzoo_env(runtime, episode, log_directory)
    try:
        return wrap_torchrl(env, seed, device)
    except BaseException:
        env.close()
        raise

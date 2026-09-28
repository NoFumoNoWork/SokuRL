"""Declare SokuRL as a BenchMARL task through its PettingZoo adapter."""
from functools import partial
from math import ceil

from benchmarl.environments.pettingzoo.common import PettingZooClass

from .torchrl_env import make_torchrl_env


class SokuTask(PettingZooClass):
    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        if continuous_actions:
            raise ValueError("SokuRL uses discrete held-key commands")
        # BenchMARL owns vector construction around this single-game factory.
        return partial(make_torchrl_env, self.config["runtime"], self.config["episode"],
                       self.config["log_directory"], seed, device)

    def supports_continuous_actions(self):
        return False

    def supports_discrete_actions(self):
        return True

    def has_state(self):
        return False

    def has_render(self, env):
        return self.config["episode"]["observation_mode"] == "image"

    def max_steps(self, env):
        episode = self.config["episode"]
        return ceil(episode["max_frames"] / episode["decision_frames"])

    @staticmethod
    def env_name():
        return "sokurl"

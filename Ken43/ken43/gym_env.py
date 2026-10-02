from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .bots import AlwaysBlockBot, select_bot_action, update_bot
from .parallel_env import Ken43ParallelEnv


class Ken43GymEnv(gym.Env):
    """Role-selectable Gymnasium facade against a configurable fixed bot."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent=None,
        *,
        learning_agent: str = "player_0",
        bootstrap_timeouts: bool = False,
        **parallel_kwargs,
    ):
        self.parallel = Ken43ParallelEnv(**parallel_kwargs)
        self.opponent = opponent or AlwaysBlockBot()
        if learning_agent not in self.parallel.possible_agents:
            raise ValueError("learning_agent must be player_0 or player_1")
        self.learning_agent = learning_agent
        self.learning_actor = self.parallel.possible_agents.index(learning_agent)
        self.opponent_actor = 1 - self.learning_actor
        self.opponent_agent = self.parallel.possible_agents[self.opponent_actor]
        self.bootstrap_timeouts = bool(bootstrap_timeouts)
        self.action_space = self.parallel.action_space(self.learning_agent)
        self.observation_space = self.parallel.observation_space(self.learning_agent)["observation"]
        self._rng = np.random.default_rng()
        self._observations = None

    @property
    def command_index(self):
        return self.parallel.command_index

    @property
    def commands(self):
        return self.parallel.commands

    def action_masks(self):
        if self._observations is None:
            return np.zeros(self.action_space.n, dtype=bool)
        return self._observations[self.learning_agent]["action_mask"].astype(bool)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        reset_opponent = getattr(self.opponent, "reset", None)
        if callable(reset_opponent):
            reset_opponent()
        self._observations, infos = self.parallel.reset(seed=seed, options=options)
        return self._observations[self.learning_agent]["observation"], infos[self.learning_agent]

    def step(self, action):
        if self._observations is None:
            raise RuntimeError("reset() must be called before step()")
        opponent_action = select_bot_action(
            self.opponent,
            self._observations[self.opponent_agent],
            self.parallel.command_index,
            self._rng,
            self.parallel,
            self.opponent_actor,
        )
        joint_actions = {
            self.learning_agent: np.asarray(action),
            self.opponent_agent: np.asarray(opponent_action),
        }
        observations, rewards, terminations, truncations, infos = self.parallel.step(
            joint_actions
        )
        self._observations = observations
        update_bot(self.opponent, infos, self.parallel, self.opponent_actor)
        terminated = terminations[self.learning_agent]
        truncated = truncations[self.learning_agent]
        if (
            truncated
            and infos[self.learning_agent].get("terminal_reason") == "max_frames"
            and not self.bootstrap_timeouts
        ):
            terminated = True
            truncated = False
            infos[self.learning_agent]["timeout_treated_as_terminal"] = True
        return (
            observations[self.learning_agent]["observation"],
            rewards[self.learning_agent],
            terminated,
            truncated,
            infos[self.learning_agent],
        )

    def render(self):
        return None

    def close(self):
        self.parallel.close()

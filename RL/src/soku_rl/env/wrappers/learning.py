"""Apply one learning contract to both PettingZoo and vector game environments."""
from collections import deque
from dataclasses import dataclass

from gymnasium import spaces
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

from soku_rl.env.encoding import AGENTS, decode_action
from soku_rl.env.wrappers.features import RELATIVE_FEATURES, health_potential, relative_features


@dataclass(frozen=True)
class LearningConfig:
    action_set: str
    relative_features: bool
    action_history: int
    health_potential_scale: float

    def __post_init__(self):
        if self.action_set not in {"full", "combat"}:
            raise ValueError("action_set must be full or combat")
        if type(self.relative_features) is not bool or type(self.action_history) is not int or self.action_history < 0:
            raise ValueError("invalid feature or action-history configuration")
        if not np.isfinite(self.health_potential_scale) or self.health_potential_scale < 0:
            raise ValueError("potential scale must be finite and nonnegative")


class LearningInterface:
    """A reversible action map and a public observation layout, without state."""
    def __init__(self, episode, config):
        self.episode, self.config = episode, config
        self.base_space = episode.space()
        buttons = range(64) if config.action_set == "full" else (0, 1, 2, 4, 8, 16, 32, 9, 10, 12)
        self.commands = tuple(axis * 64 + button for axis in range(9) for button in buttons)
        self.inverse = {value: index for index, value in enumerate(self.commands)}
        self.action_space = spaces.Discrete(len(self.commands))
        extra = RELATIVE_FEATURES * config.relative_features + 8 * config.action_history
        if episode.observation_mode == "image" and config.health_potential_scale:
            raise ValueError("health shaping cannot read image observations")
        if config.relative_features and episode.observation_mode != "state":
            raise ValueError("relative screen features require public state observations")
        if episode.observation_mode == "image" and config.action_history:
            self.observation_space = spaces.Dict({"image": self.base_space,
                "commands": spaces.Box(-1, 1, (8 * config.action_history,), np.float32)})
        elif extra:
            low = np.concatenate((self.base_space.low, np.full(extra, -1, np.float32)))
            high = np.concatenate((self.base_space.high, np.ones(extra, np.float32)))
            self.observation_space = spaces.Box(low, high, dtype=np.float32)
        else:
            self.observation_space = self.base_space

    def command(self, action):
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("learning action must be an integer")
        if not 0 <= action < len(self.commands):
            raise ValueError("learning action outside the configured vocabulary")
        return self.commands[int(action)]

    def action(self, command):
        if command not in self.inverse:
            raise ValueError("base command is unavailable in the configured action set")
        return self.inverse[command]

    def base_observation(self, observation):
        if isinstance(self.observation_space, spaces.Dict):
            if (not isinstance(observation, dict) or set(observation) != {"image", "commands"}
                    or observation["image"].shape != self.base_space.shape
                    or observation["commands"].shape != self.observation_space["commands"].shape):
                raise ValueError("learning observation does not match the image/commands layout")
            return observation["image"]
        if observation.shape != self.observation_space.shape:
            raise ValueError("learning observation has an incorrect shape")
        if observation.ndim == 1:
            return observation[:self.base_space.shape[0]]
        return observation


class LearningEpisode:
    """Keep only own submitted commands and the previous health potential."""
    def __init__(self, interface):
        self.interface = interface
        self.history = {a: deque(maxlen=interface.config.action_history) for a in AGENTS}
        self.potential = {}

    def reset(self, observations, infos):
        for agent in AGENTS:
            self.reset_agent(agent, observations[agent])
        return self._observations(observations, infos), infos

    def reset_agent(self, agent, observation):
        self.history[agent].clear()
        self.history[agent].extend([decode_action(256).inputs] * self.interface.config.action_history)
        self.potential[agent] = self._potential(observation)

    def record_command(self, agent, command):
        self.history[agent].append(decode_action(command).inputs)

    def step(self, commands, result):
        observations, rewards, terms, truncs, infos = result
        shaped, output_infos = {}, {}
        for agent in AGENTS:
            self.record_command(agent, commands[agent])
            current = 0. if terms[agent] or truncs[agent] else self._potential(observations[agent])
            difference = current - self.potential[agent]
            self.potential[agent] = current
            shaped[agent] = rewards[agent] + difference
            output_infos[agent] = infos[agent] | {"base_reward": rewards[agent], "shaping_reward": difference}
        return self._observations(observations, infos), shaped, terms, truncs, output_infos

    def _potential(self, observation):
        scale = self.interface.config.health_potential_scale
        return scale * health_potential(observation, self.interface.episode.observation_mode) if scale else 0.

    def _observations(self, observations, infos):
        return {agent: self.observation(agent, observation, infos[agent]["frame"])
                for agent, observation in observations.items()}

    def observation(self, agent, observation, frame):
        if isinstance(self.interface.observation_space, spaces.Dict):
            return {"image": observation,
                "commands": np.asarray(self.history[agent], np.float32).reshape(-1)}
        parts = [observation]
        if self.interface.config.relative_features:
            parts.append(relative_features(observation, self.interface.episode, frame))
        if self.interface.config.action_history:
            parts.append(np.asarray(self.history[agent], np.float32).reshape(-1))
        return np.concatenate(parts) if len(parts) > 1 else observation


class LearningParallelEnv(BaseParallelWrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.interface = LearningInterface(env.episode.config, config)
        self.transform = LearningEpisode(self.interface)
        self.observation_spaces = dict.fromkeys(AGENTS, self.interface.observation_space)
        self.action_spaces = dict.fromkeys(AGENTS, self.interface.action_space)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        return self.transform.reset(*self.env.reset(seed=seed, options=options))

    def step(self, actions):
        commands = {a: self.interface.command(v) for a, v in actions.items()}
        result = self.env.step(commands)
        return self.transform.step(commands, result) if result[0] else result


class LearningVectorEnv:
    def __init__(self, env, config):
        self.env, self.num_envs, self.episodes = env, env.num_envs, env.episodes
        self.possible_agents = env.possible_agents
        self.interface = LearningInterface(env.episodes[0].config, config)
        self.transforms = {slot: LearningEpisode(self.interface) for slot in env.episodes}
        self.single_action_space = self.interface.action_space
        self.single_observation_space = self.interface.observation_space

    def reset(self, seeds):
        observations, infos = self.env.reset(seeds)
        results = {slot: self.transforms[slot].reset(observations[slot], infos[slot]) for slot in seeds}
        return tuple({s: r[i] for s, r in results.items()} for i in range(2))

    def reset_matchups(self, seeds, matches):
        observations, infos = self.env.reset_matchups(seeds, matches)
        results = {slot: self.transforms[slot].reset(observations[slot], infos[slot]) for slot in seeds}
        return tuple({s: r[i] for s, r in results.items()} for i in range(2))

    def step(self, actions):
        commands = {s: {a: self.interface.command(v) for a, v in players.items()} for s, players in actions.items()}
        result = self.env.step(commands)
        results = {s: self.transforms[s].step(commands[s], tuple(value[s] for value in result)) for s in actions}
        return tuple({s: r[i] for s, r in results.items()} for i in range(5))

    def close(self):
        self.env.close()

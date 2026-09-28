"""PettingZoo simultaneous two-player episodes over an owned game backend."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from soku_rl.pixels import RGBFrame
from .encoding import AGENTS, NUM_ACTIONS, decode_action, encode_observation, observation_space
from .control import ControlConfig, DelayedControls


@dataclass(frozen=True)
class EpisodeConfig:
    max_frames: int
    history_frames: int
    decision_frames: int
    latency_frames: int
    observation_mode: str

    def __post_init__(self):
        if type(self.max_frames) is not int or self.max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        observation_space(self.history_frames)
        ControlConfig(self.decision_frames, self.latency_frames)
        if self.observation_mode not in {"image", "diagnostic_state"}:
            raise ValueError("unsupported observation mode; visible state is not implemented yet")

    def space(self):
        if self.observation_mode == "image":
            return spaces.Box(0, 255, (3 * self.history_frames + 1, 240, 320), np.uint8)
        return observation_space(self.history_frames)

    def encode(self, observation):
        if self.observation_mode == "image":
            if not isinstance(observation, RGBFrame) or (observation.width, observation.height) != (320, 240):
                raise ValueError("expected a native 320x240 RGB frame")
            return np.frombuffer(observation.pixels, np.uint8).reshape(240, 320, 3).transpose(2, 0, 1).copy()
        return encode_observation(observation, self.max_frames)


class Episode:
    """Own episode counters and observation history, never a policy."""
    def __init__(self, config):
        self.config = config
        self.history = [deque(maxlen=config.history_frames) for _ in AGENTS]
        self.ready = False
        self.ended = True
        self.frame = 0
        self.number = 0
        self.controls = DelayedControls(ControlConfig(config.decision_frames, config.latency_frames))

    def reset(self, time_step, seed):
        if time_step.frame != 0 or time_step.ended or time_step.rewards != (0, 0):
            raise RuntimeError("reset did not produce a clean ongoing frame zero")
        self.seed = seed
        self.number += 1
        self.frame = 0
        self.controls.reset()
        self.ready, self.ended = True, False
        for history, observation in zip(self.history, time_step.observations, strict=True):
            history.clear()
            if isinstance(observation, RGBFrame) and observation.frame != time_step.frame:
                raise RuntimeError("image and simulation frame do not match")
            encoded = self.config.encode(observation)
            history.extend(encoded.copy() for _ in range(self.config.history_frames))
        return self._observations(), self._infos(time_step, "ongoing")

    def actions(self, actions):
        if not self.ready or self.ended:
            raise RuntimeError("reset this episode before stepping")
        if set(actions) != set(AGENTS):
            raise ValueError("both players must submit an action in the same step")
        return tuple(decode_action(actions[agent]) for agent in AGENTS)

    def submit(self, actions):
        self.actions(actions)
        self.controls.submit(self.frame, actions)

    def inputs(self):
        return self.controls.inputs(self.frame)

    def invalidate(self):
        self.ready, self.ended = False, True
        for history in self.history:
            history.clear()

    def step(self, time_step):
        if time_step.frame != self.frame + 1:
            raise RuntimeError("backend must advance exactly one frame")
        self.frame = time_step.frame
        for history, observation in zip(self.history, time_step.observations, strict=True):
            if isinstance(observation, RGBFrame) and observation.frame != time_step.frame:
                raise RuntimeError("image and simulation frame do not match")
            history.append(self.config.encode(observation))
        terminated = time_step.terminated
        truncated = not terminated and (time_step.truncated or self.frame >= self.config.max_frames)
        self.ended = terminated or truncated
        outcome = "time_limit" if truncated else time_step.outcome.value
        return (self._observations(), dict(zip(AGENTS, time_step.rewards, strict=True)),
                dict.fromkeys(AGENTS, terminated), dict.fromkeys(AGENTS, truncated),
                self._infos(time_step, outcome))

    def _observations(self):
        result = {agent: np.concatenate(self.history[index]) for index, agent in enumerate(AGENTS)}
        if self.config.observation_mode == "image":
            for index, agent in enumerate(AGENTS):
                # Public seat identity allows parameter sharing between the two roles.
                role = np.full((1, 240, 320), index * 255, dtype=np.uint8)
                result[agent] = np.concatenate((result[agent], role))
        return result

    def _infos(self, time_step, outcome):
        return {agent: {"frame": time_step.frame, "episode": self.number,
                        "outcome": outcome,
                        "decision_frames": self.config.decision_frames,
                        "latency_frames": self.config.latency_frames}
                for agent in AGENTS}


class HisoutenParallelEnv(ParallelEnv):
    metadata = {"name": "sokurl_v0", "render_modes": [], "is_parallelizable": True}
    render_mode = None

    def __init__(self, backend, config):
        self.backend = backend
        self.episode = Episode(config)
        self.possible_agents = list(AGENTS)
        self.agents = []
        self.observation_spaces = {a: config.space() for a in AGENTS}
        self.action_spaces = {a: spaces.Discrete(NUM_ACTIONS) for a in AGENTS}
        self.np_random, self.np_random_seed = seeding.np_random(None)
        self.closed = False

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if self.closed:
            raise RuntimeError("environment is closed")
        # PettingZoo's public contract permits generic reset option dictionaries.
        # Game settings are fixed by the construction config, not by this dictionary.
        if options is not None and not isinstance(options, dict):
            raise TypeError("reset options must be a dictionary or None")
        if seed is not None:
            if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF:
                raise ValueError("seed must be in [0, 0xFFFFFFFF); the upper value is reserved")
            self.np_random, self.np_random_seed = seeding.np_random(seed)
            world_seed = seed
        else:
            world_seed = int(self.np_random.integers(0, 0xFFFFFFFF, dtype=np.uint64))
        self.episode.invalidate()
        self.agents = []
        result = self.episode.reset(self.backend.reset_slots({0: world_seed})[0], world_seed)
        self.agents = list(AGENTS)
        return result

    def step(self, actions):
        if self.closed:
            raise RuntimeError("environment is closed")
        if not self.agents and self.episode.ready and actions == {}:
            return {}, {}, {}, {}, {}
        try:
            self.episode.submit(actions)
            for _ in range(self.episode.config.decision_frames):
                result = self.episode.step(self.backend.step({0: self.episode.inputs()})[0])
                if self.episode.ended:
                    break
        except BaseException:
            self.episode.invalidate()
            self.agents = []
            raise
        if self.episode.ended:
            self.agents = []
        return result

    def close(self):
        if not self.closed:
            self.backend.close()
            self.closed = True
            self.agents = []

    def render(self):
        if self.episode.config.observation_mode != "image" or not self.episode.ready:
            raise RuntimeError("render requires a reset image environment")
        return self.episode.history[0][-1].transpose(1, 2, 0).copy()

    def state(self):
        raise NotImplementedError("the bridge does not expose the full engine state")

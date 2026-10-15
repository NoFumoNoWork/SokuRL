"""Stack consecutive public observations for training and live inference."""
from collections import deque

import numpy as np

from soku_rl.env.observation.pixels import RGBFrame
from soku_rl.env.observation.visible_state import StateObservation
from soku_rl.env.encoding import AGENTS
from soku_rl.env.observation.animation import AnimationObservation, HISTORY_OFFSETS


class ObservationHistory:
    def __init__(self, config, agents):
        if not agents or len(set(agents)) != len(agents) or any(agent not in AGENTS for agent in agents):
            raise ValueError("history requires distinct supported agent names")
        self.config = config
        self.indices = {agent: AGENTS.index(agent) for agent in agents}
        length = 13 if config.observation_mode == 'animation' else config.history_frames
        self.history = {agent: deque(maxlen=length) for agent in agents}
        self.frame = -1

    def _encode(self, frame, observations):
        if type(frame) is not int or frame < 0 or len(observations) != len(AGENTS):
            raise ValueError("a nonnegative frame and both public observations are required")
        for observation in observations:
            if isinstance(observation, (RGBFrame, StateObservation, AnimationObservation)) and observation.frame != frame:
                raise RuntimeError("image and simulation frame do not match")
        return {agent: self.config.encode(observations[index]) for agent, index in self.indices.items()}

    def reset(self, frame, observations):
        encoded = self._encode(frame, observations)
        for agent, history in self.history.items():
            value = encoded[agent]
            history.clear()
            history.extend(value.copy() for _ in range(history.maxlen))
        self.frame = frame

    def append(self, frame, observations):
        if self.frame < 0 or frame != self.frame + 1:
            raise RuntimeError("observation history requires consecutive frames after reset")
        encoded = self._encode(frame, observations)
        for agent, history in self.history.items():
            history.append(encoded[agent])
        self.frame = frame

    def clear(self):
        for history in self.history.values():
            history.clear()
        self.frame = -1

    def observations(self):
        if self.frame < 0:
            raise RuntimeError("reset observation history before reading it")
        result = {agent: np.concatenate([history[-1-offset] for offset in HISTORY_OFFSETS]
                                         if self.config.observation_mode == 'animation' else history)
                  for agent, history in self.history.items()}
        if self.config.observation_mode == "image":
            for agent, index in self.indices.items():
                role = np.full((1, 240, 320), index * 255, dtype=np.uint8)
                result[agent] = np.concatenate((result[agent], role))
        return result

    def image(self, seat):
        if (self.frame < 0 or self.config.observation_mode != "image" or type(seat) is not int
                or seat not in (0, 1) or AGENTS[seat] not in self.history):
            raise ValueError("image requires a reset image history and a valid seat")
        return self.history[AGENTS[seat]][-1].transpose(1, 2, 0).copy()

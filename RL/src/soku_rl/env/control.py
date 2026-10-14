"""Schedule joint key states against simulation time, independently of wall time."""
from collections import deque
from dataclasses import dataclass

from soku_rl.env.encoding import AGENTS, decode_action


@dataclass(frozen=True)
class ControlConfig:
    decision_frames: int
    latency_frames: int

    def __post_init__(self):
        if type(self.decision_frames) is not int or self.decision_frames < 1:
            raise ValueError("decision_frames must be a positive integer")
        if type(self.latency_frames) is not int or self.latency_frames < 0:
            raise ValueError("latency_frames must be a nonnegative integer")


class DelayedControls:
    """An observation at t can affect the transition t+latency -> t+latency+1.

    Both players submit together. A command is a held key state, not a macro.
    Reset removes all pending commands and releases every key.
    """
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.pending = deque()
        self.held = (decode_action(256), decode_action(256))
        self.next_decision = 0
        self.next_frame = 0

    def submit(self, frame, actions):
        if frame != self.next_decision or frame != self.next_frame:
            raise ValueError("submit exactly once at each decision boundary")
        if set(actions) != set(AGENTS):
            raise ValueError("both players must submit simultaneously")
        joint = tuple(decode_action(actions[a]) for a in AGENTS)
        self.pending.append((frame + self.config.latency_frames, joint))
        self.next_decision += self.config.decision_frames

    def inputs(self, frame):
        if frame != self.next_frame or frame >= self.next_decision:
            raise ValueError("advance each frame once, after submitting its decision")
        if self.pending and self.pending[0][0] == frame:
            _, self.held = self.pending.popleft()
        self.next_frame += 1
        return self.held

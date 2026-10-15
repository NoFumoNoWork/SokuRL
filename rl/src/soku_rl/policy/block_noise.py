"""Persist replacement gates without holding commands or freezing the original actor."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from soku_rl.policy.action_noise import ActionNoiseActor, ActionNoisePolicy


@dataclass(frozen=True)
class BlockActionNoisePolicy(ActionNoisePolicy):
    block_decisions: int

    def __post_init__(self):
        super().__post_init__()
        if type(self.block_decisions) is not int or self.block_decisions < 1:
            raise ValueError("block_decisions must be a positive integer")

    @property
    def fingerprint(self):
        identity = ["block-action-noise-v1", self.policy.fingerprint, self.num_actions,
                    float(self.random_probability), self.block_decisions]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    def _wrap(self, actor, seed):
        gate, actions = np.random.SeedSequence(seed).spawn(2)
        return BlockActionNoiseActor(actor, np.random.default_rng(gate),
            np.random.default_rng(actions), self.num_actions, self.random_probability,
            self.block_decisions, 0, False)


@dataclass
class BlockActionNoiseActor(ActionNoiseActor):
    block_decisions: int
    remaining_decisions: int
    replace_block: bool

    def act(self, observation):
        # Only the source-selection gate persists. God still sees each frame,
        # and uniform commands are independently sampled within a noisy block.
        action = self.actor.act(observation)
        if self.remaining_decisions == 0:
            self.replace_block = self.gate_rng.random() < self.random_probability
            self.remaining_decisions = self.block_decisions
        self.remaining_decisions -= 1
        if self.replace_block:
            return int(self.action_rng.integers(self.num_actions))
        return action

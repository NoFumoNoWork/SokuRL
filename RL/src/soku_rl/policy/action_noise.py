"""Frozen opponents with explicit per-decision uniform action replacement."""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from soku_rl.policy.base import PlayActor, Policy


@dataclass(frozen=True)
class ActionNoisePolicy(Policy):
    name: str
    policy: object
    num_actions: int
    random_probability: float

    def __post_init__(self):
        p = self.random_probability
        if (type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                or type(self.num_actions) is not int or self.num_actions < 1):
            raise ValueError("action noise requires a probability in [0, 1] and positive action count")

    @property
    def fingerprint(self):
        identity = ["action-noise-v1", self.policy.fingerprint, self.num_actions,
                    float(self.random_probability)]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    def _wrap(self, actor, seed):
        gate, actions = np.random.SeedSequence(seed).spawn(2)
        return ActionNoiseActor(actor, np.random.default_rng(gate),
            np.random.default_rng(actions), self.num_actions, self.random_probability)

    def spawn(self, seed):
        return self._wrap(self.policy.spawn(seed), seed)

    def spawn_with_seeds(self, noise_seed, policy_seed):
        return self._wrap(self.policy.spawn(policy_seed), noise_seed)

    def spawn_play(self, seed):
        instance = self.policy.spawn_play(seed)
        return PlayActor(self._wrap(instance.actor, seed), instance.reset_each_round)


@dataclass
class ActionNoiseActor:
    actor: object
    gate_rng: object
    action_rng: object
    num_actions: int
    random_probability: float

    def act(self, observation):
        return self.act_with_label(observation)[1]

    def act_with_label(self, observation):
        """Return the underlying label, executed action and replacement decision."""
        # Advance the original controller on every observation, including replaced
        # decisions. Otherwise its timers, recurrent memory and RNG would freeze.
        action = self.actor.act(observation)
        if self.gate_rng.random() < self.random_probability:
            return action, int(self.action_rng.integers(self.num_actions)), True
        return action, action, False

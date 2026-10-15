"""Immutable policies and mixtures that select one policy per episode."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from soku_rl.policy.base import Policy, RLPolicy, RulePolicy


@dataclass(frozen=True)
class UniformPolicy(RulePolicy):
    name: str
    num_actions: int

    @property
    def fingerprint(self):
        return f"uniform-{self.num_actions}-v1"

    def spawn(self, seed):
        return UniformEpisode(np.random.default_rng(seed), self.num_actions)


@dataclass
class UniformEpisode:
    rng: object
    num_actions: int

    def act(self, observation):
        return int(self.rng.integers(self.num_actions))


class PPOPolicy(RLPolicy):
    def __init__(self, name, model, path):
        self.name, self.model, self.path = name, model, path
        self.model.policy.set_training_mode(False)
        self.fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()

    def spawn(self, seed):
        return PPOEpisode(self.model, np.random.default_rng(seed))


@dataclass
class PPOEpisode:
    model: object
    rng: object

    def act(self, observation):
        import torch
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            distribution = self.model.policy.get_distribution(tensor).distribution
            probabilities = distribution.probs[0].detach().cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(len(probabilities), p=probabilities))


class MixturePolicy(Policy):
    """Select one frozen population member for the whole episode."""
    def __init__(self, name, members, probabilities, identity):
        weights = np.asarray(probabilities, dtype=np.float64)
        if (not members or weights.shape != (len(members),) or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.isclose(weights.sum(), 1.)):
            raise ValueError("population weights must form a probability distribution")
        self.name, self.members, self.fingerprint = name, tuple(members), identity
        self.probabilities = weights / weights.sum()
        self.probabilities.setflags(write=False)

    def _select(self, seed):
        rng = np.random.default_rng(seed)
        member = self.members[int(rng.choice(len(self.members), p=self.probabilities))]
        return member, int(rng.integers(0, 0xFFFFFFFF))

    def spawn(self, seed):
        member, actor_seed = self._select(seed)
        return member.spawn(actor_seed)

    def spawn_play(self, seed):
        member, actor_seed = self._select(seed)
        return member.spawn_play(actor_seed)


@dataclass(frozen=True)
class SeatPolicies:
    name: str
    roles: tuple

    def __post_init__(self):
        if len(self.roles) != 2:
            raise ValueError("two seat policies are required")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps([p.fingerprint for p in self.roles]).encode()).hexdigest()

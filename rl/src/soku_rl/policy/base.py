"""Policy definitions create independent actors for each player and episode."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


class Actor(Protocol):
    def act(self, observation): ...


@dataclass(frozen=True)
class PlayActor:
    """A selected actor and its required memory boundary in an original match."""
    actor: Actor
    reset_each_round: bool

    def act(self, observation):
        return self.actor.act(observation)


class PrivilegedPlayActor(PlayActor):
    """Also accept the complete decoded frame, without a dense tensor roundtrip."""

    def act_observation(self, observation):
        return self.actor.act_observation(observation)


class Policy(ABC):
    name: str
    fingerprint: str

    @abstractmethod
    def spawn(self, seed: int) -> Actor:
        """Create fresh memory and a private random stream for one episode."""

    def spawn_play(self, seed):
        """Use training episode boundaries unless a policy requires continuous play."""
        return PlayActor(self.spawn(seed), True)


class RLPolicy(Policy):
    """A policy whose action distribution is represented by learned parameters."""


class RulePolicy(Policy):
    """A policy whose actions are produced by explicit rules or source scripts."""

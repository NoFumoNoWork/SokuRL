"""Frozen DQN policies act greedily; training exploration is never deployed."""
from dataclasses import dataclass

from soku_rl.policy.population import PPOPolicy


class DQNPolicy(PPOPolicy):
    def spawn(self, seed):
        return DQNEpisode(self.model)


@dataclass
class DQNEpisode:
    model: object

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

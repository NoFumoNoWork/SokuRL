"""Immutable policy snapshots and vector payoff sampling for a strategy population."""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .env.encoding import AGENTS, NUM_ACTIONS


@dataclass(frozen=True)
class UniformPolicy:
    name: str

    @property
    def fingerprint(self):
        return "uniform-576-v1"

    def spawn(self, seed):
        return UniformEpisode(np.random.default_rng(seed))


@dataclass
class UniformEpisode:
    rng: object

    def act(self, observation):
        return int(self.rng.integers(NUM_ACTIONS))


class PPOPolicy:
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
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            distribution = self.model.policy.get_distribution(tensor).distribution
            probabilities = distribution.probs[0].detach().cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(NUM_ACTIONS, p=probabilities))


class PopulationEvaluator:
    """Evaluate role-specific policies; never swap player populations implicitly."""
    def __init__(self, env, seed, timeout_payoff):
        if timeout_payoff != "zero_at_horizon":
            raise ValueError("PSRO requires the declared finite-horizon payoff")
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.records = []

    def evaluate(self, policies, num_episodes):
        if len(policies) != 2 or type(num_episodes) is not int or num_episodes < 1:
            raise ValueError("two policies and a positive episode count required")
        total = np.zeros(2)
        for start in range(0, num_episodes, self.env.num_envs):
            count = min(self.env.num_envs, num_episodes - start)
            seeds = {s: int(self.rng.integers(0, 0xFFFFFFFF)) for s in range(count)}
            private_seeds = {s: tuple(int(self.rng.integers(0, 0xFFFFFFFF)) for _ in AGENTS)
                             for s in seeds}
            actors = {s: tuple(p.spawn(z) for p, z in zip(policies, private_seeds[s], strict=True))
                      for s in seeds}
            observations, _ = self.env.reset(seeds)
            returns = {s: np.zeros(2) for s in seeds}
            while observations:
                actions = {s: {a: actors[s][i].act(obs[a]) for i, a in enumerate(AGENTS)}
                           for s, obs in observations.items()}
                next_obs, rewards, terms, truncs, infos = self.env.step(actions)
                for slot in list(next_obs):
                    returns[slot] += [rewards[slot][a] for a in AGENTS]
                    if terms[slot][AGENTS[0]] or truncs[slot][AGENTS[0]]:
                        total += returns[slot]
                        self.records.append({
                            "policies": [p.name for p in policies],
                            "fingerprints": [p.fingerprint for p in policies],
                            "world_seed": seeds[slot], "policy_seeds": private_seeds[slot],
                            "returns": returns[slot].tolist(),
                            "final": infos[slot][AGENTS[0]],
                        })
                        del next_obs[slot]
                observations = next_obs
        return total / num_episodes

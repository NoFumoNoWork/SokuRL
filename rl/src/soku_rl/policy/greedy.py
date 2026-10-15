"""Explicit deterministic variants of frozen PPO policies, with distinct identities."""
import hashlib

import torch

from soku_rl.policy.base import RLPolicy
from soku_rl.policy.population import PPOPolicy
from soku_rl.policy.recurrent import RecurrentPPOPolicy


class GreedyPPOPolicy(RLPolicy):
    def __init__(self, name, policy):
        if not isinstance(policy, PPOPolicy):
            raise ValueError("greedy inference requires a frozen PPO policy")
        self.name, self.model = name, policy.model
        self.recurrent = isinstance(policy, RecurrentPPOPolicy)
        self.fingerprint = hashlib.sha256(
            f"greedy-ppo-v1:{policy.fingerprint}".encode()).hexdigest()

    def spawn(self, seed):
        # The seed remains part of the evaluation contract, but no action RNG is used.
        return GreedyPPOEpisode(self.model, self.recurrent)


class GreedyPPOEpisode:
    def __init__(self, model, recurrent):
        self.model, self.recurrent = model, recurrent
        if recurrent:
            policy = model.policy
            self.states = tuple(torch.zeros(policy.lstm_hidden_state_shape,
                device=policy.device) for _ in range(2))
            self.start = torch.ones(1, device=policy.device)

    @torch.inference_mode()
    def act(self, observation):
        policy = self.model.policy
        tensor, _ = policy.obs_to_tensor(observation)
        if self.recurrent:
            distribution, self.states = policy.get_distribution(tensor, self.states, self.start)
            self.start.zero_()
        else:
            distribution = policy.get_distribution(tensor)
        if not torch.isfinite(distribution.distribution.probs).all():
            raise RuntimeError("greedy policy produced non-finite probabilities")
        return int(distribution.mode()[0].item())

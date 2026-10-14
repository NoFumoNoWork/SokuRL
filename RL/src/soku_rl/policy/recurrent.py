"""Keep SB3 recurrent-policy memory and action RNG private to each episode."""
import numpy as np
import torch

from soku_rl.policy.population import PPOPolicy


class RecurrentPPOPolicy(PPOPolicy):
    def spawn(self, seed):
        return RecurrentEpisode(self.model, seed)


class RecurrentEpisode:
    def __init__(self, model, seed):
        self.model = model
        self.rng = np.random.default_rng(seed)
        policy = model.policy
        self.states = tuple(torch.zeros(policy.lstm_hidden_state_shape, device=policy.device) for _ in range(2))
        self.start = torch.ones(1, device=policy.device)

    @torch.inference_mode()
    def act(self, observation):
        tensor, _ = self.model.policy.obs_to_tensor(observation)
        distribution, self.states = self.model.policy.get_distribution(tensor, self.states, self.start)
        self.start.zero_()
        probabilities = distribution.distribution.probs[0].cpu().numpy().astype(np.float64)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("recurrent policy produced non-finite probabilities")
        probabilities /= probabilities.sum()
        return int(self.rng.choice(len(probabilities), p=probabilities))

"""Static categorical masking without renumbering existing logical actions."""
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


class CardMaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, masked_actions=(), **kwargs):
        self.masked_actions=tuple(masked_actions)
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError('static card mask requires categorical actions')
        blocked=torch.zeros(self.action_space.n,dtype=torch.bool,device=self.device)
        if (not self.masked_actions or len(set(self.masked_actions))!=len(self.masked_actions)
                or any(type(i) is not int or not 0<=i<self.action_space.n for i in self.masked_actions)
                or len(self.masked_actions)==self.action_space.n):
            raise ValueError('invalid static action mask')
        blocked[list(self.masked_actions)]=True
        self.register_buffer('_blocked_actions',blocked,persistent=False)

    def _get_action_dist_from_latent(self, latent_pi):
        logits=self.action_net(latent_pi).masked_fill(self._blocked_actions,float('-inf'))
        return self.action_dist.proba_distribution(action_logits=logits)

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters() | dict(masked_actions=self.masked_actions)

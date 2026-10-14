"""Tie direction/button logits while retaining the complete categorical command space."""
import math

from gymnasium import spaces
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class DirectionButtonHead(nn.Module):
    def __init__(self, latent_dim, button_probability, orthogonal):
        super().__init__()
        if (type(button_probability) not in (int, float) or not math.isfinite(button_probability)
                or not 0 < button_probability < 1):
            raise ValueError("factorized button_probability must be strictly between zero and one")
        self.factors = nn.Linear(latent_dim, 15)
        if orthogonal:
            nn.init.orthogonal_(self.factors.weight, gain=.01)
        nn.init.zeros_(self.factors.bias)
        with torch.no_grad():
            self.factors.bias[9:] = math.log(button_probability / (1 - button_probability))
        commands = torch.arange(576)
        self.register_buffer("directions", commands // 64, persistent=False)
        self.register_buffer("buttons", ((commands[:, None] % 64 >> torch.arange(6)) & 1).float(), persistent=False)

    def forward(self, latent):
        factors = self.factors(latent)
        # Normalizing these 576 logits yields a 9-way direction categorical
        # times six Bernoullis. Sampling and PPO likelihood still use the same
        # full categorical distribution, including all multi-button commands.
        return factors[:, :9][:, self.directions] + factors[:, 9:] @ self.buttons.T


class FactorizedHeadMixin:
    def __init__(self, *args, factor_button_probability, **kwargs):
        self.factor_button_probability = factor_button_probability
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        if not isinstance(self.action_space, spaces.Discrete) or self.action_space.n != 576:
            raise ValueError("factorized policy requires all 576 logical commands")
        super()._build(lr_schedule)
        self.action_net = DirectionButtonHead(self.mlp_extractor.latent_dim_pi,
            self.factor_button_probability, self.ortho_init)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters() | {"factor_button_probability": self.factor_button_probability}


class FactorizedActorCriticPolicy(FactorizedHeadMixin, ActorCriticPolicy):
    """Shared direction/button head with the upstream feedforward policy."""

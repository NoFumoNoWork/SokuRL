"""Learn a per-frame mixture of repeating the last command and choosing anew."""
import math

from gymnasium import spaces
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

from soku_rl.rl.combat_features import CombatPrivilegedFeatures


class ActionContextFeatures(CombatPrivilegedFeatures):
    def __init__(self, observation_space, history_frames, object_features, player_features, features_dim):
        super().__init__(observation_space, history_frames, object_features, player_features, features_dim)
        if observation_space.shape[0] - self.base_width < 8:
            raise ValueError("persistent policy requires the last eight command components")
        self._features_dim = features_dim + 8

    def forward(self, observations):
        return torch.cat((super().forward(observations), observations[:, -8:]), dim=1)


class CommandContextMlp(nn.Module):
    def __init__(self, features_dim, net_arch, activation_fn, device):
        super().__init__()
        self.network = MlpExtractor(features_dim - 8, net_arch, activation_fn, device)
        self.latent_dim_pi = self.network.latent_dim_pi + 8
        self.latent_dim_vf = self.network.latent_dim_vf

    def forward_actor(self, features):
        return torch.cat((self.network.forward_actor(features[:, :-8]), features[:, -8:]), dim=1)

    def forward_critic(self, features):
        return self.network.forward_critic(features[:, :-8])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class RepeatMixtureHead(nn.Module):
    def __init__(self, action_net, probability):
        super().__init__()
        self.fresh = action_net
        self.gate = nn.Linear(action_net.in_features - 8, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(probability / (1 - probability)))
        self.register_buffer("button_weights", 2 ** torch.arange(6), persistent=False)

    @property
    def bias(self):
        # The existing fresh-action button prior applies only to the new choice.
        return self.fresh.bias

    def forward(self, latent):
        command = latent[:, -8:]
        previous = (((command[:, 0] + 1) * 3 + command[:, 1] + 1) * 64
                    + (command[:, 2:] * self.button_weights).sum(1)).long()
        gate = self.gate(latent[:, :-8])
        fresh = F.log_softmax(self.fresh(latent), dim=1) + F.logsigmoid(-gate)
        repeat = torch.full_like(fresh, -torch.inf).scatter(1, previous[:, None], F.logsigmoid(gate))
        # These are log probabilities of the full mixture, so PPO evaluates the
        # same likelihood used to sample. Every frame can select any command.
        return torch.logaddexp(fresh, repeat)


class PersistentActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, repeat_probability, **kwargs):
        if (type(repeat_probability) not in (int, float) or not math.isfinite(repeat_probability)
                or not 0 < repeat_probability < 1):
            raise ValueError("repeat_probability must be strictly between zero and one")
        self.repeat_probability = repeat_probability
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        if (not isinstance(self.action_space, spaces.Discrete) or self.action_space.n != 576
                or not isinstance(self.features_extractor, ActionContextFeatures)):
            raise ValueError("persistent policy requires full actions and ActionContextFeatures")
        self.mlp_extractor = CommandContextMlp(self.features_dim, self.net_arch, self.activation_fn, self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = RepeatMixtureHead(self.action_net, self.repeat_probability)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters() | {"repeat_probability": self.repeat_probability}

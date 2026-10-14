"""A per-frame repeat/new-command mixture using the shared recurrent PPO policy."""
import math

from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.policies import BaseModel
import torch
from torch import nn
from torch.nn import functional as F


def previous_commands(observations, button_weights):
    command = observations[:, -8:]
    return (((command[:, 0] + 1) * 3 + command[:, 1] + 1) * 64
            + (command[:, 2:] * button_weights).sum(1)).long()


class PersistentRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, repeat_probability, **kwargs):
        if (type(repeat_probability) not in (int, float) or not math.isfinite(repeat_probability)
                or not 0 < repeat_probability < 1):
            raise ValueError("repeat_probability must be strictly between zero and one")
        self.repeat_probability = repeat_probability
        super().__init__(*args, **kwargs)
        if (not isinstance(self.action_space, spaces.Discrete) or self.action_space.n != 576
                or not isinstance(self.observation_space, spaces.Box)
                or len(self.observation_space.shape) != 1 or self.observation_space.shape[0] < 8):
            raise ValueError("recurrent persistence requires 576 commands and command-history observations")
        self.register_buffer("button_weights", 2 ** torch.arange(6, device=self.device), persistent=False)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Gate parameters have deterministic initial values. Do not consume the
        # RNG stream that the upstream constructor uses for its LSTM weights.
        with torch.random.fork_rng(devices=[]):
            self.repeat_gate = nn.Linear(self.mlp_extractor.latent_dim_pi, 1)
        nn.init.zeros_(self.repeat_gate.weight)
        nn.init.constant_(self.repeat_gate.bias, math.log(self.repeat_probability / (1 - self.repeat_probability)))
        # RecurrentActorCriticPolicy builds its optimizer after the LSTMs, so it
        # includes this gate as well as the upstream actor and critic parameters.

    def _mixed_distribution(self, observations, latent):
        previous = previous_commands(observations, self.button_weights)
        gate = self.repeat_gate(latent)
        fresh = F.log_softmax(self.action_net(latent), dim=1) + F.logsigmoid(-gate)
        repeat = torch.full_like(fresh, -torch.inf).scatter(1, previous[:, None], F.logsigmoid(gate))
        # The public distribution cache is the full mixture, including for BC.
        return self.action_dist.proba_distribution(action_logits=torch.logaddexp(fresh, repeat))

    def _latents(self, observations, lstm_states, episode_starts):
        features = self.extract_features(observations)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            states_vf = tuple(state.detach() for state in states_pi)
        else:
            latent_vf = self.critic(vf_features)
            states_vf = states_pi
        return (self.mlp_extractor.forward_actor(latent_pi), self.mlp_extractor.forward_critic(latent_vf),
                RNNStates(states_pi, states_vf))

    def forward(self, observations, lstm_states, episode_starts, *args, **kwargs):
        latent_pi, latent_vf, states = self._latents(observations, lstm_states, episode_starts)
        distribution = self._mixed_distribution(observations, latent_pi)
        # Delegate the existing optional deterministic argument to SB3 unchanged.
        actions = distribution.get_actions(*args, **kwargs)
        return (actions.reshape((-1, *self.action_space.shape)), self.value_net(latent_vf),
                distribution.log_prob(actions), states)

    def evaluate_actions(self, observations, actions, lstm_states, episode_starts):
        latent_pi, latent_vf, _ = self._latents(observations, lstm_states, episode_starts)
        distribution = self._mixed_distribution(observations, latent_pi)
        return self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, observations, lstm_states, episode_starts):
        features = BaseModel.extract_features(self, observations, self.pi_features_extractor)
        latent, states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        return self._mixed_distribution(observations, self.mlp_extractor.forward_actor(latent)), states

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters() | {
            "repeat_probability": self.repeat_probability,
            "lstm_hidden_size": self.lstm_actor.hidden_size,
            "n_lstm_layers": self.lstm_actor.num_layers,
            "shared_lstm": self.shared_lstm,
            "enable_critic_lstm": self.enable_critic_lstm,
            "lstm_kwargs": dict(self.lstm_kwargs),
        }

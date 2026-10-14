"""Express a full recurrent categorical action head in the fighter's own facing."""
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
import torch


def left_facing(observations, index):
    facing = observations[:, index] * 4294967296. + observations[:, index + 1] * 65536.
    # Zero-filled recurrent padding uses identity; its loss is masked by SB3.
    return facing < 0


class FacingRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, facing_index, **kwargs):
        if type(facing_index) is not int or facing_index < 0:
            raise ValueError("facing_index must locate an encoded fighter direction")
        self.facing_index = facing_index
        super().__init__(*args, **kwargs)
        if (not isinstance(self.action_space, spaces.Discrete) or self.action_space.n != 576
                or not isinstance(self.observation_space, spaces.Box)
                or len(self.observation_space.shape) != 1
                or facing_index + 2 > self.observation_space.shape[0]):
            raise ValueError("facing actions require 576 commands and an encoded direction pair")
        commands = torch.arange(576, device=self.device)
        # Horizontal groups contain 3 vertical directions * 64 button chords.
        mirror = (2 - commands // 192) * 192 + commands % 192
        self.register_buffer("facing_commands", commands, persistent=False)
        self.register_buffer("mirrored_commands", mirror, persistent=False)

    def _left_facing(self, observations):
        return left_facing(observations, self.facing_index)

    def _remap_actions(self, observations, actions):
        mirror = self.mirrored_commands[actions.long()]
        return torch.where(self._left_facing(observations), mirror, actions)

    def _absolute_distribution(self, observations):
        permutation = torch.where(self._left_facing(observations)[:, None],
                                  self.mirrored_commands, self.facing_commands)
        logits = self.action_dist.distribution.logits.gather(1, permutation)
        # Keep the public cache in absolute game coordinates too: recurrent BC
        # and auxiliary losses consume action_dist after forward/evaluate_actions.
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, observations, *args, **kwargs):
        actions, values, log_prob, states = super().forward(observations, *args, **kwargs)
        actions = self._remap_actions(observations, actions)
        self._absolute_distribution(observations)
        return actions, values, log_prob, states

    def evaluate_actions(self, observations, actions, lstm_states, episode_starts):
        relative = self._remap_actions(observations, actions)
        result = super().evaluate_actions(observations, relative, lstm_states, episode_starts)
        self._absolute_distribution(observations)
        return result

    def get_distribution(self, observations, lstm_states, episode_starts):
        _, states = super().get_distribution(observations, lstm_states, episode_starts)
        return self._absolute_distribution(observations), states

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters() | {
            "facing_index": self.facing_index,
            "lstm_hidden_size": self.lstm_actor.hidden_size,
            "n_lstm_layers": self.lstm_actor.num_layers,
            "shared_lstm": self.shared_lstm,
            "enable_critic_lstm": self.enable_critic_lstm,
            "lstm_kwargs": dict(self.lstm_kwargs),
        }

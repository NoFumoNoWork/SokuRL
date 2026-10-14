"""Use the shared full-command factorized head with upstream recurrent PPO."""
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from soku_rl.rl.factorized_policy import FactorizedHeadMixin


class FactorizedRecurrentActorCriticPolicy(FactorizedHeadMixin, RecurrentActorCriticPolicy):
    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters() | {
            "lstm_hidden_size": self.lstm_actor.hidden_size,
            "n_lstm_layers": self.lstm_actor.num_layers,
            "shared_lstm": self.shared_lstm,
            "enable_critic_lstm": self.enable_critic_lstm,
            "lstm_kwargs": dict(self.lstm_kwargs),
        }

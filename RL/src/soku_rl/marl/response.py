"""Train PSRO responses through the shared PPO learner."""
from pathlib import Path

import numpy as np

from soku_rl.policy.population import PPOPolicy
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
from soku_rl.rl.learner import create_learner, snapshot, learner_kind


class PPOResponseOracle:
    def __init__(self, env, config, device, seed, directory):
        if config["initialization"] not in {"fresh", "parent_weights"}:
            raise ValueError("PPO response initialization must be fresh or parent_weights")
        self.env, self.config, self.device = env, config, device
        self.rng = np.random.default_rng(seed)
        self.directory = Path(directory)
        self.responses = 0

    def __call__(self, game, training_parameters, strategy_sampler, using_joint_strategies):
        if using_joint_strategies or game.num_players() != 2:
            raise ValueError("this oracle supports two role-specific marginal populations")
        results = [[], []]
        for player, requests in enumerate(training_parameters):
            for request in requests:
                if request["current_player"] != player:
                    raise ValueError("oracle request has inconsistent player IDs")
                opponents = request["total_policies"][1 - player]
                probabilities = request["probabilities_of_playing_policies"][1 - player]
                seed = int(self.rng.integers(0, 2**31))
                view = OpponentMixtureVecEnv(self.env, player, opponents, probabilities, seed)
                try:
                    model, _ = create_learner(view, self.env.interface, self.config,
                        {"kind": "fresh"}, self.device, seed)
                    # Copy only policy parameters. New optimizer and schedule belong
                    # to this response; old population snapshots remain unchanged.
                    parent = request["policy"]
                    if self.config["initialization"] == "parent_weights" and isinstance(parent, PPOPolicy):
                        model.policy.load_state_dict(parent.model.policy.state_dict())
                    model.learn(total_timesteps=self.config["timesteps_per_response"])
                    self.responses += 1
                    name = f"{learner_kind(self.config)}-p{player}-response-{self.responses}"
                    path = self.directory / (name + ".zip")
                    model.save(path)
                    results[player].append(snapshot(name, model, path))
                finally:
                    view.close()
        return results

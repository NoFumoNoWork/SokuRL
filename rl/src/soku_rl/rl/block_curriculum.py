"""Reuse the long-term EMA controller with correlated opponent-noise gates."""
from soku_rl.policy.block_noise import BlockActionNoisePolicy
from soku_rl.rl.curriculum import AdaptiveActionNoise


class AdaptiveBlockActionNoise(AdaptiveActionNoise):
    kind = "adaptive_block_action_noise"

    def __init__(self, config, opponents, probabilities, num_actions):
        if ("block_decisions" not in config or type(config["block_decisions"]) is not int
                or config["block_decisions"] < 1):
            raise ValueError("block_decisions must be a positive integer")
        feedback = {key: value for key, value in config.items() if key != "block_decisions"}
        super().__init__(feedback, opponents, probabilities, num_actions)
        self.config = dict(config)

    def spawn(self, opponent, seed):
        state = self.states[opponent.name]
        noisy = BlockActionNoisePolicy(opponent.name, opponent, self.num_actions,
            state["random_probability"], self.config["block_decisions"])
        context = {"random_probability": state["random_probability"],
            "controller_episodes_at_start": state["episodes"],
            "training_opponent_fingerprint": noisy.fingerprint,
            "block_decisions": self.config["block_decisions"]}
        return noisy.spawn(seed), {"curriculum": context}

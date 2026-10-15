"""Keep the response character fixed while sampling its seat and opponent."""
from dataclasses import asdict

from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv


class MatchupMixtureVecEnv(OpponentMixtureVecEnv):
    def __init__(self, env, player, opponents, probabilities, seed, learner, setups):
        self.learner = PlayerSetup(**learner)
        self.setups = [PlayerSetup(**setup) for setup in setups]
        if len(self.setups) != len(opponents):
            raise ValueError("each opponent requires its own character setup")
        super().__init__(env, player, opponents, probabilities, seed)

    def _reset_game(self, seeds):
        matches = {}
        for slot in seeds:
            opponent = self.setups[self.opponent_indices[slot]]
            pair = (self.learner, opponent) if self.players[slot] == 0 else (opponent, self.learner)
            matches[slot] = MatchConfig(*pair)
            self.episode_context[slot]["match"] = asdict(matches[slot])
        return self.env.reset_matchups(seeds, matches)

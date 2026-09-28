"""Use public SB3 PPO as a response oracle over the two-player vector game."""
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from .env.encoding import AGENTS
from .population import PPOPolicy


class OpponentMixtureVecEnv(VecEnv):
    """Training view for one response; the underlying environment stays two-player.

    Opponent identities and private RNG state stay fixed for an entire episode.
    This view borrows the vector environment and does not own its worker.
    """
    def __init__(self, env, player, opponents, probabilities, seed):
        if player not in (0, 1) or not opponents:
            raise ValueError("a player role and opponent population are required")
        weights = np.asarray(probabilities, dtype=np.float64)
        if (weights.shape != (len(opponents),) or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.isclose(weights.sum(), 1)):
            raise ValueError("opponent probabilities must form a distribution")
        self.env, self.player = env, player
        self.opponents, self.probabilities = opponents, weights / weights.sum()
        self.rng = np.random.default_rng(seed)
        self.render_mode = None
        self.actors = {}
        self.observations = {}
        self.pending = None
        self.closed = False
        self.returns = np.zeros(env.num_envs)
        self.lengths = np.zeros(env.num_envs, dtype=np.int64)
        super().__init__(env.num_envs, env.single_observation_space, env.single_action_space)

    def _reset_slots(self, seeds):
        observations, infos = self.env.reset(seeds)
        for slot in seeds:
            opponent = self.opponents[int(self.rng.choice(len(self.opponents), p=self.probabilities))]
            self.actors[slot] = opponent.spawn(int(self.rng.integers(0, 0xFFFFFFFF)))
            self.returns[slot] = self.lengths[slot] = 0
            self.reset_infos[slot] = infos[slot][AGENTS[self.player]]
        self.observations.update(observations)

    def reset(self):
        if self.closed or self.pending is not None:
            raise RuntimeError("cannot reset a closed or pending PPO view")
        if any(options for options in self._options):
            raise ValueError("reset options are unsupported")
        seeds = {i: int(self._seeds[i]) if self._seeds[i] is not None else
                 int(self.rng.integers(0, 0xFFFFFFFF)) for i in range(self.num_envs)}
        self._reset_slots(seeds)
        self._reset_seeds()
        self._reset_options()
        return self._stack()

    def _stack(self):
        return np.stack([self.observations[i][AGENTS[self.player]] for i in range(self.num_envs)])

    def step_async(self, actions):
        if self.closed or self.pending is not None or not self.observations:
            raise RuntimeError("PPO view is not ready for step_async")
        if np.asarray(actions).shape != (self.num_envs,):
            raise ValueError("one learner action per environment is required")
        self.pending = np.asarray(actions).copy()

    def step_wait(self):
        if self.pending is None:
            raise RuntimeError("step_async must precede step_wait")
        actions = {i: {
            AGENTS[self.player]: self.pending[i],
            AGENTS[1 - self.player]: self.actors[i].act(self.observations[i][AGENTS[1 - self.player]])
        } for i in range(self.num_envs)}
        self.pending = None
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        values = np.asarray([rewards[i][AGENTS[self.player]] for i in range(self.num_envs)], dtype=np.float32)
        dones = np.asarray([terms[i][AGENTS[self.player]] or truncs[i][AGENTS[self.player]]
                            for i in range(self.num_envs)])
        self.returns += values
        self.lengths += 1
        output_infos, seeds = [], {}
        for i in range(self.num_envs):
            info = dict(infos[i][AGENTS[self.player]])
            info["episode_id"] = info.pop("episode")
            info["source_truncated"] = bool(truncs[i][AGENTS[self.player]])
            # Both PPO and PSRO use the explicitly chosen finite-horizon game.
            # Setting this true would make SB3 add an unrequested bootstrap value.
            info["TimeLimit.truncated"] = False
            if dones[i]:
                info["terminal_observation"] = obs[i][AGENTS[self.player]].copy()
                info["episode"] = {"r": float(self.returns[i]), "l": int(self.lengths[i])}
                seeds[i] = int(self.rng.integers(0, 0xFFFFFFFF))
            output_infos.append(info)
        self.observations = obs
        if seeds:
            self._reset_slots(seeds)
        return self._stack(), values, dones, output_infos

    def close(self):
        self.closed = True

    def get_attr(self, attr_name, indices=None):
        value = getattr(self, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("runtime environment attributes are fixed")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("use the two-player environment for runtime control")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class PPOResponseOracle:
    def __init__(self, env, config, device, seed, directory):
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
                    policy_type = "CnnPolicy" if len(view.observation_space.shape) == 3 else "MlpPolicy"
                    model = PPO(policy_type, view, device=self.device, seed=seed,
                                **self.config["ppo"])
                    # Copy only policy parameters. New optimizer and schedule belong
                    # to this response; old population snapshots remain unchanged.
                    parent = request["policy"]
                    if isinstance(parent, PPOPolicy):
                        model.policy.load_state_dict(parent.model.policy.state_dict())
                    model.learn(total_timesteps=self.config["timesteps_per_response"])
                    self.responses += 1
                    name = f"ppo-p{player}-response-{self.responses}"
                    path = self.directory / (name + ".zip")
                    model.save(path)
                    results[player].append(PPOPolicy(name, model, path))
                finally:
                    view.close()
        return results

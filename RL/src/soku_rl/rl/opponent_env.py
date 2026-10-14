"""Use public SB3 PPO as a response oracle over the two-player vector game."""
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from soku_rl.env.encoding import AGENTS
from soku_rl.rl.curriculum import FixedOpponentSchedule


class OpponentMixtureVecEnv(VecEnv):
    """Training view for one response; the underlying environment stays two-player.

    Opponent identities and private RNG state stay fixed for an entire episode.
    This view borrows the vector environment and does not own its worker.
    """
    def __init__(self, env, player, opponents, probabilities, seed):
        if player not in (0, 1, "random", "balanced") or not opponents:
            raise ValueError("a player role and opponent population are required")
        if player == "balanced" and (env.num_envs < 2 or env.num_envs % 2):
            raise ValueError("balanced seats require a positive even number of environments")
        weights = np.asarray(probabilities, dtype=np.float64)
        if (weights.shape != (len(opponents),) or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.isclose(weights.sum(), 1)):
            raise ValueError("opponent probabilities must form a distribution")
        self.env, self.player = env, player
        self.opponents, self.probabilities = opponents, weights / weights.sum()
        self.rng = np.random.default_rng(seed)
        self.render_mode = None
        self.actors = {}
        self.players = {}
        self.opponent_indices = {}
        self.episode_context = {}
        self.observations = {}
        self.pending = None
        self.closed = False
        self.curriculum = FixedOpponentSchedule()
        self.returns = np.zeros(env.num_envs)
        self.base_returns = np.zeros(env.num_envs)
        self.lengths = np.zeros(env.num_envs, dtype=np.int64)
        super().__init__(env.num_envs, env.single_observation_space, env.single_action_space)

    def _reset_slots(self, seeds):
        for slot in seeds:
            if self.player in ("random", "balanced"):
                # Consume the same draw in both modes so identical reset
                # schedules retain identical world/opponent RNG streams.
                sampled = int(self.rng.integers(0, 2))
                self.players[slot] = slot % 2 if self.player == "balanced" else sampled
            else:
                self.players[slot] = self.player
            index = int(self.rng.choice(len(self.opponents), p=self.probabilities))
            self.opponent_indices[slot] = index
            opponent = self.opponents[index]
            policy_seed = int(self.rng.integers(0, 0xFFFFFFFF))
            self.actors[slot], curriculum_context = self.curriculum.spawn(opponent, policy_seed)
            self.episode_context[slot] = {"world_seed": seeds[slot], "opponent_seed": policy_seed,
                "player": self.players[slot], "opponent": opponent.name, "opponent_fingerprint": opponent.fingerprint}
            self.episode_context[slot].update(curriculum_context)
            self.returns[slot] = self.base_returns[slot] = self.lengths[slot] = 0
        observations, infos = self._reset_game(seeds)
        for slot in seeds:
            self.reset_infos[slot] = infos[slot][AGENTS[self.players[slot]]]
        self.observations.update(observations)

    def _reset_game(self, seeds):
        return self.env.reset(seeds)

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
        values = [self.observations[i][AGENTS[self.players[i]]] for i in range(self.num_envs)]
        if isinstance(self.observation_space, spaces.Dict):
            return {key: np.stack([value[key] for value in values]) for key in self.observation_space.spaces}
        return np.stack(values)

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
            AGENTS[self.players[i]]: self.pending[i],
            AGENTS[1 - self.players[i]]: self.actors[i].act(self.observations[i][AGENTS[1 - self.players[i]]])
        } for i in range(self.num_envs)}
        self.pending = None
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        values = np.asarray([rewards[i][AGENTS[self.players[i]]] for i in range(self.num_envs)], dtype=np.float32)
        dones = np.asarray([terms[i][AGENTS[self.players[i]]] or truncs[i][AGENTS[self.players[i]]]
                            for i in range(self.num_envs)])
        self.returns += values
        self.base_returns += [infos[i][AGENTS[self.players[i]]]["base_reward"] for i in range(self.num_envs)]
        self.lengths += 1
        output_infos, seeds = [], {}
        for i in range(self.num_envs):
            player = self.players[i]
            info = dict(infos[i][AGENTS[player]])
            info["episode_id"] = info.pop("episode")
            info["source_truncated"] = bool(truncs[i][AGENTS[player]])
            # Both PPO and PSRO use the explicitly chosen finite-horizon game.
            # Setting this true would make SB3 add an unrequested bootstrap value.
            info["TimeLimit.truncated"] = False
            if dones[i]:
                terminal = obs[i][AGENTS[player]]
                info["terminal_observation"] = ({k: v.copy() for k, v in terminal.items()}
                    if isinstance(self.observation_space, spaces.Dict) else terminal.copy())
                info["episode"] = {"r": float(self.returns[i]), "l": int(self.lengths[i])}
                info["training_context"] = self.episode_context[i] | {"base_return": float(self.base_returns[i])}
                event = self.curriculum.observe(self.episode_context[i], info)
                if event:
                    info["curriculum_event"] = event
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

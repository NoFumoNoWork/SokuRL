"""Explicit partial reset and joint stepping for multiple two-player episodes."""
from gymnasium import spaces
from .encoding import AGENTS, NUM_ACTIONS, observation_space
from .hisouten_env import Episode


class TwoPlayerVectorEnv:
    """Outer keys are environment IDs; inner keys are player IDs.

    No automatic reset. Terminal observations remain from the ended episode.
    A subset of slots can step while the rest stay paused without losing state.
    """
    def __init__(self, backend, num_envs, config):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.backend = backend
        self.num_envs = num_envs
        self.episodes = {i: Episode(config) for i in range(num_envs)}
        self.possible_agents = AGENTS
        self.single_observation_space = observation_space(config.history_frames)
        self.single_action_space = spaces.Discrete(NUM_ACTIONS)
        self.closed = False

    def _check_slots(self, slots):
        if self.closed:
            raise RuntimeError("vector environment is closed")
        if not slots or any(type(s) is not int or s not in self.episodes for s in slots):
            raise ValueError("a nonempty set of valid environment IDs is required")

    def reset(self, seeds):
        self._check_slots(seeds)
        if any(type(s) is not int or not 0 <= s < 0xFFFFFFFF for s in seeds.values()):
            raise ValueError("each seed must be in [0, 0xFFFFFFFF)")
        for slot in seeds:
            self.episodes[slot].invalidate()
        states = self.backend.reset_slots(seeds)
        if set(states) != set(seeds):
            raise RuntimeError("backend returned incorrect reset slots")
        results = {s: self.episodes[s].reset(states[s], seeds[s]) for s in seeds}
        return ({s: r[0] for s, r in results.items()}, {s: r[1] for s, r in results.items()})

    def step(self, actions):
        self._check_slots(actions)
        # Validate every joint decision before changing any episode's queue.
        for slot, action in actions.items():
            self.episodes[slot].actions(action)
        try:
            for slot, action in actions.items():
                self.episodes[slot].submit(action)
            results = {}
            active = set(actions)
            for _ in range(next(iter(self.episodes.values())).config.decision_frames):
                joint = {s: self.episodes[s].inputs() for s in sorted(active)}
                states = self.backend.step(joint)
                if set(states) != active:
                    raise RuntimeError("backend returned incorrect step slots")
                results.update({s: self.episodes[s].step(states[s]) for s in active})
                active = {s for s in active if not self.episodes[s].ended}
                if not active:
                    break
        except BaseException:
            for slot in actions:
                self.episodes[slot].invalidate()
            raise
        return tuple({s: r[i] for s, r in results.items()} for i in range(5))

    def close(self):
        if not self.closed:
            self.backend.close()
            self.closed = True

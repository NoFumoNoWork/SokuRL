"""Run a rule or learned policy on consecutive frames with private memory."""
from dataclasses import replace

from soku_rl.env.encoding import AGENTS
from soku_rl.env.observation.privileged import PrivilegedObservation
from soku_rl.env.observation.diagnostic import Observation
from soku_rl.env.observation_history import ObservationHistory
from soku_rl.env.wrappers.learning import LearningEpisode
from soku_rl.policy.base import PrivilegedPlayActor


class LivePolicy:
    """Keep policy memory, observation history and own commands in one instance.

    The caller supplies observations for the configured track and submits returned
    commands through the local input transport. Command history records issued
    intents, not proof of engine execution; the caller logs transport outcomes.
    Skipped decision windows do not invent actions or advance model memory.
    The training time-limit feature saturates, but the original match continues.
    """

    def __init__(self, policy, interface, seat):
        if type(seat) is not int or seat not in (0, 1):
            raise ValueError("live policy requires a local seat")
        if interface.episode.observation_mode not in ("state", "image", "privileged_state", "diagnostic_state"):
            raise ValueError("live policy requires public or complete privileged observations")
        self.policy, self.interface = policy, interface
        self.agent = AGENTS[seat]
        self.history = ObservationHistory(interface.episode, (self.agent,))
        self.features = LearningEpisode(interface)
        self.active = False
        self.prepared = {}
        self.frame = -1

    def prepare(self, seed):
        """Create independent actor memory while the game is still in its menus."""
        if self.active:
            raise RuntimeError("cannot prepare an actor during an active round")
        if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF:
            raise ValueError("round seed must be a supported uint32")
        if self.prepared and seed not in self.prepared:
            raise RuntimeError("prepared actor seed differs from the next round")
        if not self.prepared:
            self.prepared[seed] = self.policy.spawn_play(seed)

    def start_round(self, frame, observations, seed):
        if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF:
            raise ValueError("round seed must be a supported uint32")
        self.active = False
        self.origin = self.next_decision = frame
        if self.prepared and seed not in self.prepared:
            raise RuntimeError("prepared actor seed differs from the next round")
        self.actor = self.prepared.pop(seed) if self.prepared else self.policy.spawn_play(seed)
        self._observe(frame, observations, True)
        self.active = True

    def _observe(self, frame, observations, reset):
        if type(frame) is not int or frame < 0 or len(observations) != len(AGENTS):
            raise ValueError("a nonnegative frame and both observations are required")
        current = self._round_observations(frame, observations)
        if isinstance(self.actor, PrivilegedPlayActor):
            if self.interface.episode.observation_mode != "privileged_state":
                raise ValueError("decoded script input requires complete privileged observations")
            self.observation = current[AGENTS.index(self.agent)]
        elif reset:
            self.history.reset(frame, current)
            self.features.reset_agent(self.agent, self.history.observations()[self.agent])
        else:
            self.history.append(frame, current)
        self.frame = frame

    def _round_observations(self, frame, observations):
        if self.interface.episode.observation_mode == "diagnostic_state":
            if any(not isinstance(value, Observation) or value.frame != frame for value in observations):
                raise ValueError("diagnostic observation must identify the exact supplied frame")
            return tuple(replace(value, frame=min(frame - self.origin, self.interface.episode.max_frames))
                         for value in observations)
        if self.interface.episode.observation_mode != "privileged_state":
            return observations
        for value in observations:
            if not isinstance(value, PrivilegedObservation) or value.world["frame"] != frame:
                raise ValueError("privileged observation must identify the exact supplied frame")
        # Only the synthetic episode frame is relative to this round. Preserve
        # the engine's battle clock and every original script input unchanged.
        return tuple(replace(value, world=value.world | {"frame": frame - self.origin})
                     for value in observations)

    @property
    def reset_each_round(self):
        if not self.active:
            raise RuntimeError("start a policy before reading its memory boundary")
        return self.actor.reset_each_round

    @property
    def decision_due(self):
        return self.active and self.frame == self.next_decision

    def observe(self, frame, observations):
        if not self.active:
            raise RuntimeError("start a round before observing frames")
        if self.decision_due:
            raise RuntimeError("consume the pending decision before advancing history")
        if frame != self.frame + 1:
            raise RuntimeError("observation history requires consecutive frames after reset")
        self._observe(frame, observations, False)

    def act(self):
        if not self.decision_due:
            raise RuntimeError("act exactly once at each decision boundary")
        if isinstance(self.actor, PrivilegedPlayActor):
            action = self.actor.act_observation(self.observation)
        else:
            base = self.history.observations()[self.agent]
            elapsed = min(self.frame-self.origin, self.interface.episode.max_frames)
            observation = self.features.observation(self.agent, base, elapsed)
            action = self.actor.act(observation)
        command = self.interface.command(action)
        self.features.record_command(self.agent, command)
        self.next_decision += self.interface.episode.decision_frames
        return command

    def skip_decision(self):
        if not self.decision_due:
            raise RuntimeError("skip only an unconsumed decision boundary")
        if self.interface.episode.observation_mode == "privileged_state":
            raise RuntimeError("complete privileged policies cannot skip a decision frame")
        self.next_decision += self.interface.episode.decision_frames

    def stop(self):
        self.active = False
        self.history.clear()
        self.observation = ()
        self.frame = -1

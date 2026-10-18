"""PettingZoo simultaneous two-player episodes over an owned game backend."""
from dataclasses import dataclass, asdict
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from soku_rl.env.observation.pixels import RGBFrame
from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.env.observation.visible_state import StateObservation, STATE_FEATURES
from soku_rl.env.encoding import AGENTS, NUM_ACTIONS, decode_action, encode_observation, observation_space
from soku_rl.env.control import ControlConfig, DelayedControls
from soku_rl.env.observation_history import ObservationHistory
from soku_rl.env.match import LEGACY_MATCH, MatchConfig
from soku_rl.env.observation.privileged import encode_privileged
from soku_rl.env.observation.memory_schema import PRIVILEGED_FEATURES
from soku_rl.env.combat_metrics import CombatMetrics
from soku_rl.env.input_metrics import InputMetrics
from soku_rl.env.observation.animation import AnimationObservation, frame_width, HISTORY_OFFSETS
from soku_rl.env.reward import CombatReward, CombatRewardConfig


@dataclass(frozen=True)
class EpisodeConfig:
    max_frames: int
    history_frames: int
    decision_frames: int
    latency_frames: int
    observation_mode: str
    visibility: VisibilityConfig
    match: MatchConfig
    animation_object_slots: int = 16
    combat_reward: dict | None = None
    animation_world_coordinates: bool = False
    shared_teacher_context: bool = False

    def __post_init__(self):
        if isinstance(self.match, dict):
            object.__setattr__(self, "match", MatchConfig(**self.match))
        if not isinstance(self.match, MatchConfig):
            raise TypeError("match must be a MatchConfig")
        if type(self.max_frames) is not int or self.max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        observation_space(self.history_frames)
        ControlConfig(self.decision_frames, self.latency_frames)
        if isinstance(self.visibility, dict):
            object.__setattr__(self, "visibility", VisibilityConfig(**self.visibility))
        if not isinstance(self.visibility, VisibilityConfig):
            raise TypeError("visibility must be a VisibilityConfig or its configuration dictionary")
        frame_width(self.animation_object_slots)
        if type(self.shared_teacher_context) is not bool or (self.shared_teacher_context and
                (self.observation_mode!='animation' or not self.animation_world_coordinates or
                 self.animation_object_slots!=16 or self.decision_frames!=1 or self.latency_frames!=0)):
            raise ValueError('shared context requires animation/world16/1F/zero latency')
        if type(self.animation_world_coordinates) is not bool or (self.animation_world_coordinates and self.observation_mode != 'animation'):
            raise ValueError('world coordinates require explicit animation mode')
        if self.observation_mode == "animation" and self.history_frames != len(HISTORY_OFFSETS):
            raise ValueError('animation mode requires eight sparse past-frame samples')
        if self.observation_mode not in {"image", "state", "diagnostic_state", "privileged_state", "animation"}:
            raise ValueError("unsupported observation mode")
        if self.combat_reward is not None:
            CombatRewardConfig(**self.combat_reward)
            if self.observation_mode != 'animation' or self.decision_frames!=1 or self.latency_frames!=0:
                raise ValueError('native combat reward requires animation/1F/zero latency')

    def backend_observation(self):
        result = {"mode": self.observation_mode, "visibility": asdict(self.visibility),
                  "match": asdict(self.match), "max_frames": self.max_frames}
        if self.observation_mode == 'animation':
            if self.shared_teacher_context:
                result['shared_teacher_context'] = True
            result['animation_object_slots'] = self.animation_object_slots
            if self.combat_reward is not None:
                reward=CombatRewardConfig(**self.combat_reward)
                result['passive_policy']=dict(pursuit_distance=reward.pursuit_distance,
                    minimum_progress=reward.pursuit_minimum_progress)
        return result

    @classmethod
    def from_dict(cls, values):
        """Read the current schema or migrate the fixed selections of schema 1."""
        return cls(**({"match": LEGACY_MATCH} | values))

    def space(self):
        if self.observation_mode == 'animation':
            return spaces.Box(-np.inf, np.inf, (frame_width(self.animation_object_slots, self.animation_world_coordinates)*8,), np.float32)
        if self.observation_mode == "privileged_state":
            return spaces.Box(-np.inf, np.inf, (PRIVILEGED_FEATURES * self.history_frames,), np.float32)
        if self.observation_mode == "image":
            return spaces.Box(0, 255, (3 * self.history_frames + 1, 240, 320), np.uint8)
        if self.observation_mode == "state":
            return spaces.Box(-1, 1, (STATE_FEATURES * self.history_frames,), np.float32)
        return observation_space(self.history_frames)

    def encode(self, observation):
        if self.observation_mode == 'animation':
            if not isinstance(observation, AnimationObservation):
                raise TypeError('expected privileged animation-proxy observation')
            core = tuple(observation.values)
            if len(core) != frame_width(self.animation_object_slots):
                raise ValueError('invalid core animation frame')
            world = tuple(observation.world_positions) if self.animation_world_coordinates else ()
            values = np.asarray(core + world, np.float32)
            if values.shape != (frame_width(self.animation_object_slots, self.animation_world_coordinates),) or not np.isfinite(values).all():
                raise ValueError('invalid animation-proxy frame')
            return values
        if self.observation_mode == "privileged_state":
            return encode_privileged(observation)
        if self.observation_mode == "image":
            if not isinstance(observation, RGBFrame) or (observation.width, observation.height) != (320, 240):
                raise ValueError("expected a native 320x240 RGB frame")
            return np.frombuffer(observation.pixels, np.uint8).reshape(240, 320, 3).transpose(2, 0, 1).copy()
        if self.observation_mode == "state":
            if not isinstance(observation, StateObservation):
                raise TypeError("expected a filtered public state observation")
            values = np.asarray(observation.values, dtype=np.float32)
            if not np.isfinite(values).all() or (values < -1).any() or (values > 1).any():
                raise ValueError("public state observation exceeds declared bounds")
            return values
        return encode_observation(observation, self.max_frames)


class Episode:
    """Own episode counters and observation history, never a policy."""
    def __init__(self, config):
        self.config = config
        self.observation_history = ObservationHistory(config, AGENTS)
        self.ready = False
        self.ended = True
        self.frame = 0
        self.number = 0
        self.combat_metrics = CombatMetrics()
        self.input_metrics = InputMetrics()
        self.controls = DelayedControls(ControlConfig(config.decision_frames, config.latency_frames))
        self.reward = CombatReward(CombatRewardConfig(**config.combat_reward)) if config.combat_reward is not None else None

    def reset(self, time_step, seed):
        if time_step.frame != 0 or time_step.ended or time_step.rewards != (0, 0):
            raise RuntimeError("reset did not produce a clean ongoing frame zero")
        self.seed = seed
        self.number += 1
        self.frame = 0
        self.controls.reset()
        self.combat_metrics.reset(time_step.observations[0])
        self.input_metrics.reset()
        self.ready, self.ended = True, False
        self.observation_history.reset(time_step.frame, time_step.observations)
        if self.reward is not None:
            self.reward.reset(time_step)
        return self.observation_history.observations(), self._infos(time_step, "ongoing")

    def actions(self, actions):
        if not self.ready or self.ended:
            raise RuntimeError("reset this episode before stepping")
        if set(actions) != set(AGENTS):
            raise ValueError("both players must submit an action in the same step")
        return tuple(decode_action(actions[agent]) for agent in AGENTS)

    def submit(self, actions):
        self.actions(actions)
        self.controls.submit(self.frame, actions)

    def inputs(self):
        return self.controls.inputs(self.frame)

    def invalidate(self):
        self.ready, self.ended = False, True
        self.observation_history.clear()

    def step(self, time_step, *, materialize=True):
        if time_step.frame != self.frame + 1:
            raise RuntimeError("backend must advance exactly one frame")
        self.frame = time_step.frame
        self.combat_metrics.step(time_step.observations[0])
        self.input_metrics.step(self.controls.held)
        self.observation_history.append(time_step.frame, time_step.observations)
        terminated = time_step.terminated
        truncated = not terminated and (time_step.truncated or self.frame >= self.config.max_frames)
        self.ended = terminated or truncated
        outcome = "time_limit" if truncated else time_step.outcome.value
        rewards = time_step.rewards
        if self.reward is not None:
            rewards, parts = self.reward.step(time_step,truncated=truncated)
            time_step.diagnostics['combat_reward_parts'] = parts
        if not materialize and not self.ended:
            return None
        return (self.observation_history.observations(), dict(zip(AGENTS, rewards, strict=True)),
                dict.fromkeys(AGENTS, terminated), dict.fromkeys(AGENTS, truncated),
                self._infos(time_step, outcome))

    def _infos(self, time_step, outcome):
        return {agent: {"frame": time_step.frame, "episode": self.number,
                        "outcome": outcome,
                        "decision_frames": self.config.decision_frames,
                        "latency_frames": self.config.latency_frames,
                        **({'combat_reward_parts':time_step.diagnostics['combat_reward_parts'][seat]}
                           if 'combat_reward_parts' in time_step.diagnostics else {}),
                        **({'combat_reward_summary':self.reward.totals[seat].copy()}
                           if self.reward is not None else {}),
                        "combat_metrics": self.combat_metrics.snapshot(seat),
                        **({"combat_events": time_step.diagnostics.get('combat_events', ())}
                           if self.config.observation_mode == 'animation' else {}),
                        **({"native_combat_events": time_step.diagnostics['native_combat_events']}
                           if 'native_combat_events' in time_step.diagnostics else {}),
                        **({'shared_native_context':time_step.diagnostics['shared_native_context'][seat],
                            'segment':time_step.diagnostics['segment']}
                           if self.config.shared_teacher_context else {}),
                        **({"input_metrics": self.input_metrics.snapshot(seat)} if self.ended else {})}
                for seat, agent in enumerate(AGENTS)}


class HisoutenParallelEnv(ParallelEnv):
    metadata = {"name": "sokurl_v0", "render_modes": [], "is_parallelizable": True}
    render_mode = None

    def __init__(self, backend, config):
        self.backend = backend
        self.episode = Episode(config)
        self.possible_agents = list(AGENTS)
        self.agents = []
        self.observation_spaces = {a: config.space() for a in AGENTS}
        self.action_spaces = {a: spaces.Discrete(NUM_ACTIONS) for a in AGENTS}
        self.np_random, self.np_random_seed = seeding.np_random(None)
        self.closed = False

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if self.closed:
            raise RuntimeError("environment is closed")
        # PettingZoo's public contract permits generic reset option dictionaries.
        # Game settings are fixed by the construction config, not by this dictionary.
        if options is not None and not isinstance(options, dict):
            raise TypeError("reset options must be a dictionary or None")
        if seed is not None:
            if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF:
                raise ValueError("seed must be in [0, 0xFFFFFFFF); the upper value is reserved")
            self.np_random, self.np_random_seed = seeding.np_random(seed)
            world_seed = seed
        else:
            world_seed = int(self.np_random.integers(0, 0xFFFFFFFF, dtype=np.uint64))
        self.episode.invalidate()
        self.agents = []
        result = self.episode.reset(self.backend.reset_slots({0: world_seed})[0], world_seed)
        self.agents = list(AGENTS)
        return result

    def step(self, actions):
        if self.closed:
            raise RuntimeError("environment is closed")
        if not self.agents and self.episode.ready and actions == {}:
            return {}, {}, {}, {}, {}
        try:
            self.episode.submit(actions)
            for _ in range(self.episode.config.decision_frames):
                result = self.episode.step(self.backend.step({0: self.episode.inputs()})[0])
                if self.episode.ended:
                    break
        except BaseException:
            self.episode.invalidate()
            self.agents = []
            raise
        if self.episode.ended:
            self.agents = []
        return result

    def close(self):
        if not self.closed:
            self.backend.close()
            self.closed = True
            self.agents = []

    def render(self):
        if self.episode.config.observation_mode != "image" or not self.episode.ready:
            raise RuntimeError("render requires a reset image environment")
        return self.episode.observation_history.image(0)

    def state(self):
        raise NotImplementedError("the bridge does not expose the full engine state")

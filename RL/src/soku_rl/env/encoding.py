"""Fixed observation tensors and lossless discrete logical key encoding."""
from dataclasses import astuple, dataclass
import numpy as np
from gymnasium import spaces

AGENTS = ("player_0", "player_1")
NUM_ACTIONS = 3 * 3 * 64
MAX_PROJECTILES = 64
FRAME_FEATURES = 19 + MAX_PROJECTILES * 5
FIGHTER_SCALES = np.asarray([1280, 1280, 10000, 1, 1000, 1, 60, 19, 1], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Decision:
    inputs: tuple[int, int, int, int, int, int, int, int]
    rule: str

def decode_action(action):
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
        raise ValueError("action must be an integer")
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"action must be in [0, {NUM_ACTIONS})")
    axes, buttons = divmod(int(action), 64)
    horizontal, vertical = divmod(axes, 3)
    return Decision((horizontal - 1, vertical - 1,
                     *((buttons >> bit) & 1 for bit in range(6))), "learned_action")


def encode_action(inputs):
    if len(inputs) != 8 or any(v not in (-1, 0, 1) for v in inputs[:2]):
        raise ValueError("invalid logical input axes")
    if any(type(v) is not int for v in inputs) or any(v not in (0, 1) for v in inputs[2:]):
        raise ValueError("invalid logical input buttons")
    return ((inputs[0] + 1) * 3 + inputs[1] + 1) * 64 + sum(
        value << bit for bit, value in enumerate(inputs[2:]))


def encode_observation(observation, horizon):
    if len(observation.enemy_projectiles) > MAX_PROJECTILES:
        raise ValueError("projectile observation exceeds the declared space")
    values = np.zeros(FRAME_FEATURES, dtype=np.float32)
    values[0] = observation.frame / horizon
    for index, fighter in enumerate((observation.player, observation.opponent)):
        values[1 + index * 9:10 + index * 9] = np.asarray(astuple(fighter)) / FIGHTER_SCALES
    for index, projectile in enumerate(observation.enemy_projectiles):
        start = 19 + 5 * index
        values[start:start + 5] = (1, projectile.x / 1280, projectile.y / 1280,
                                  projectile.speed_x / 100, projectile.speed_y / 100)
    if observation.frame < 0 or not np.isfinite(values).all():
        raise ValueError("invalid numeric observation")
    return values


def observation_space(history_frames):
    if type(history_frames) is not int or history_frames < 1:
        raise ValueError("history_frames must be a positive integer")
    return spaces.Box(-np.inf, np.inf, (FRAME_FEATURES * history_frames,), np.float32)

"""Derive learning features and health potentials from public observations only."""
import numpy as np

from soku_rl.env.encoding import FRAME_FEATURES
from soku_rl.env.observation.visible_state import STATE_FEATURES
from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, PLAYER_WIDTH, PRIVILEGED_FEATURES, WORLD_NAMES)
from soku_rl.env.observation.privileged import decode_values


RELATIVE_FEATURES = 37
HP_INDICES = np.asarray([len(WORLD_NAMES) + seat * PLAYER_WIDTH + FIGHTER_NAMES.index("hp")
                        for seat in (0, 1)])


def health_potential(observation, mode):
    if mode == "privileged_state":
        last = observation[-PRIVILEGED_FEATURES:]
        if last.shape != (PRIVILEGED_FEATURES,) or not np.isfinite(last).all():
            raise ValueError("invalid privileged observation shape or values")
        # Decode both base-65536 parts in float64, as the full decoder does.
        # Reward shaping does not need object dictionaries or collision boxes.
        hp = decode_values(last.reshape(-1, 2)[HP_INDICES])
        return float((hp[0] - hp[1]) / 10000.)
    if mode == "state":
        last = observation[-STATE_FEATURES:]
        return float(last[5] - last[13])
    if mode == "diagnostic_state":
        last = observation[-FRAME_FEATURES:]
        return float(last[3] - last[12])
    raise ValueError("health shaping requires numeric health gauges")


def relative_features(observation, episode, frame):
    """Return screen differences, visible motion, nearest objects and a clock.

    Motion is a difference between visible screen samples, not engine velocity.
    Camera motion and quantization remain in these values.
    """
    if episode.observation_mode != "state":
        raise ValueError("relative screen features require public state")
    history = observation.reshape(episode.history_frames, STATE_FEATURES)
    first, last = history[0], history[-1]
    own, enemy = last[:8], last[8:16]
    visible = own[0] == 1 and enemy[0] == 1
    dx, dy = enemy[1:3] - own[1:3] if visible else (0., 0.)
    features = [float(visible), dx, dy, np.hypot(dx, dy) / np.sqrt(2)]
    for start in (0, 8):
        valid = first[start] == 1 and last[start] == 1
        delta = last[start + 1:start + 3] - first[start + 1:start + 3] if valid else (0., 0.)
        features.extend((float(valid), *delta))
    for start in (16, 208):
        objects = last[start:start + 192].reshape(64, 3)
        objects = objects[objects[:, 0] == 1] if own[0] == 1 else objects[:0]
        relative = objects[:, 1:3] - own[1:3]
        order = np.argsort((relative ** 2).sum(-1), kind="stable")[:4]
        for index in order:
            features.extend((1., *relative[index]))
        features.extend([0.] * (3 * (4 - len(order))))
    features.extend((float(own[5] - enemy[5]), float(own[6] - enemy[6]), frame / episode.max_frames))
    result = np.asarray(features, np.float32)
    if result.shape != (RELATIVE_FEATURES,) or not np.isfinite(result).all():
        raise ValueError("invalid derived features")
    return result

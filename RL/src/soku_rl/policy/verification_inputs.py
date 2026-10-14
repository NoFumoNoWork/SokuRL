"""Bounded deployment probes with changing object counts and complete tensor layouts."""
import numpy as np

from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, FIGHTER_WIDTH, MAX_OBJECTS, OBJECT_WIDTH, PLAYER_WIDTH, RAW_WIDTH, WORLD_NAMES)
from soku_rl.env.observation.privileged import encode_values


def verification_observation(interface, rng, step):
    shape = interface.observation_space.shape
    if interface.episode.observation_mode != "privileged_state":
        return rng.uniform(-1, 1, size=shape).astype(np.float32)
    frames = []
    for history in range(interface.episode.history_frames):
        raw = np.zeros(RAW_WIDTH, np.float64)
        raw[:len(WORLD_NAMES)] = [step, step * 16, 0, 0, 0, 0, 0]
        for seat in (0, 1):
            count = (0, 1, 2, 31, 128, MAX_OBJECTS)[(step + history + seat) % 6]
            if step % 8 == 0:
                count = 0
            elif step % 8 == 7:
                count = MAX_OBJECTS
            start = len(WORLD_NAMES) + seat * PLAYER_WIDTH
            raw[start:start + FIGHTER_WIDTH] = rng.integers(0, 32, size=FIGHTER_WIDTH)
            for name, value in {"obj_n": count, "hp": 10000 - step % 10000, "rei": 5000,
                    "rmax": 5000, "x": 300 + 400 * seat,
                    "dir": (1 - 2 * seat) * (1 - 2 * ((step + history) % 2)),
                    "address": 0xFFFFFFFF - step}.items():
                raw[start + FIGHTER_NAMES.index(name)] = value
            objects = raw[start + FIGHTER_WIDTH:start + FIGHTER_WIDTH + count * OBJECT_WIDTH]
            objects[:] = rng.integers(-32, 1024, size=objects.size)
        frames.append(encode_values(raw).reshape(-1))
    base = np.concatenate(frames)
    # Own command history contains two axes and six binary buttons per command.
    extra = np.zeros(shape[0] - base.size, np.float32)
    if extra.size:
        commands = extra.reshape(-1, 8)
        commands[:, :2] = rng.integers(-1, 2, size=commands[:, :2].shape)
        commands[:, 2:] = rng.integers(0, 2, size=commands[:, 2:].shape)
    return np.concatenate((base, extra))

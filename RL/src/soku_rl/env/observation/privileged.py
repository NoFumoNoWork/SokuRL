"""Lossless common numeric observations for original Lua rules and learned policies."""
from dataclasses import dataclass

import numpy as np

from .memory_schema import (FIGHTER_NAMES, FIGHTER_WIDTH, MAX_BOXES, MAX_OBJECTS,
    OBJECT_NAMES, OBJECT_WIDTH, PLAYER_WIDTH, PRIVILEGED_FEATURES, RAW_WIDTH, WORLD_NAMES)


@dataclass(frozen=True)
class PrivilegedObservation:
    world: dict
    players: tuple


def write_entity(target, offset, entity, names):
    target[offset:offset + len(names)] = [entity[name] for name in names]
    offset += len(names)
    for key in ("hitarea", "attackarea"):
        boxes = entity[key]
        if len(boxes) > MAX_BOXES or entity[key + "_n"] != len(boxes):
            raise ValueError("invalid collision box count")
        for index, box in enumerate(boxes):
            target[offset + index * 4:offset + index * 4 + 4] = box
        offset += MAX_BOXES * 4
    return offset


def encode_privileged(observation):
    if not isinstance(observation, PrivilegedObservation):
        raise TypeError("expected a complete privileged observation")
    if len(observation.players) != 2:
        raise ValueError("complete privileged observations require both players")
    result = np.zeros((RAW_WIDTH, 2), dtype=np.float32)
    result[:len(WORLD_NAMES)] = encode_values(np.asarray([observation.world[name] for name in WORLD_NAMES], np.float64))
    for player, entity in enumerate(observation.players):
        offset = len(WORLD_NAMES) + player * PLAYER_WIDTH
        count = len(entity["objects"])
        if count != entity["obj_n"] or count > MAX_OBJECTS:
            raise ValueError("object observation exceeds declared space")
        # Preserve the fixed tensor layout, but do no arithmetic on empty slots.
        raw = np.zeros(FIGHTER_WIDTH + count * OBJECT_WIDTH, dtype=np.float64)
        cursor = write_entity(raw, 0, entity, FIGHTER_NAMES)
        for name, width in (("cards", 10), ("skills", 16), ("special", 28), ("keys", 10), ("deck", 20)):
            values = entity[name]
            if len(values) != width:
                raise ValueError(f"privileged {name} field has incorrect length")
            raw[cursor:cursor + width] = values
            cursor += width
        for index, obj in enumerate(entity["objects"]):
            write_entity(raw, FIGHTER_WIDTH + index * OBJECT_WIDTH, obj, OBJECT_NAMES)
        result[offset:offset + len(raw)] = encode_values(raw)
    return result.reshape(-1)


def encode_values(raw):
    if not np.isfinite(raw).all():
        raise ValueError("privileged observation contains non-finite fields")
    # Two base-65536 parts preserve uint32 flags/addresses as well as float32
    # fields. A single float32 would silently discard low bits of large flags.
    high = np.trunc(raw / 65536.)
    result = np.column_stack((high / 65536., (raw - high * 65536.) / 65536.)).astype(np.float32)
    if not np.array_equal(result[:, 0].astype(np.float64) * 4294967296.
                          + result[:, 1].astype(np.float64) * 65536., raw):
        raise ValueError("privileged observation cannot be represented losslessly")
    return result


def decode_values(parts):
    return parts[:, 0].astype(np.float64) * 4294967296. + parts[:, 1].astype(np.float64) * 65536.


def read_entity(raw, offset, names):
    entity = dict(zip(names, raw[offset:offset + len(names)].tolist(), strict=True))
    offset += len(names)
    for key in ("hitarea", "attackarea"):
        count = int(entity[key + "_n"])
        if not 0 <= count <= MAX_BOXES:
            raise ValueError("invalid encoded collision box count")
        entity[key] = tuple(tuple(raw[offset + i * 4:offset + i * 4 + 4]) for i in range(count))
        offset += MAX_BOXES * 4
    return entity, offset


def decode_privileged(values):
    if values.shape != (PRIVILEGED_FEATURES,) or not np.isfinite(values).all():
        raise ValueError("invalid privileged observation shape or values")
    parts = values.reshape(-1, 2)
    players = []
    for player in (0, 1):
        offset = len(WORLD_NAMES) + player * PLAYER_WIDTH
        raw = decode_values(parts[offset:offset + FIGHTER_WIDTH])
        entity, cursor = read_entity(raw, 0, FIGHTER_NAMES)
        for name, width in (("cards", 10), ("skills", 16), ("special", 28), ("keys", 10), ("deck", 20)):
            entity[name] = tuple(raw[cursor:cursor + width])
            cursor += width
        count = int(entity["obj_n"])
        if not 0 <= count <= MAX_OBJECTS:
            raise ValueError("invalid encoded object count")
        objects = decode_values(parts[offset + FIGHTER_WIDTH:offset + FIGHTER_WIDTH + count * OBJECT_WIDTH])
        entity["objects"] = tuple(read_entity(objects, i * OBJECT_WIDTH,
                                               OBJECT_NAMES)[0] for i in range(count))
        players.append(entity)
    world = decode_values(parts[:len(WORLD_NAMES)])
    return PrivilegedObservation(dict(zip(WORLD_NAMES, world, strict=True)), tuple(players))

"""Privileged native-animation proxies filtered by existing visual metadata."""
from dataclasses import dataclass

import numpy as np

from soku_rl.env.observation.visibility import visible_entities, quantize_gauge, quantize_spirit

HISTORY_OFFSETS = (12, 9, 6, 4, 3, 2, 1, 0)
TOKEN_WIDTH = 10
GLOBAL_WIDTH = 5


def frame_width(slots, world_coordinates=False):
    if type(slots) is not int or not 0 <= slots <= 64:
        raise ValueError('animation object slots must be in [0,64]')
    if type(world_coordinates) is not bool:
        raise ValueError('world_coordinates must be boolean')
    return 2 * TOKEN_WIDTH + slots * TOKEN_WIDTH + GLOBAL_WIDTH + (2 * (2+slots) if world_coordinates else 0)


@dataclass(frozen=True, slots=True)
class AnimationObservation:
    frame: int
    values: tuple[float, ...]
    world_positions: tuple[float, ...] = ()


def observe_animation(raw, render, visibility, slots=16, horizon=7200):
    """No pointers, collision flags or hidden-object categories enter the output."""
    frame_width(slots)
    players = (raw.p1, raw.p2)
    groups = (raw.p1Objects, raw.p2Objects)
    counts = (raw.p1ObjectCount, raw.p2ObjectCount)
    if raw.p1ObjectOverflow or raw.p2ObjectOverflow:
        raise RuntimeError('animation/raw object association incomplete: Bridge overflow')
    if tuple(counts) != tuple(map(len, render.objects)):
        raise RuntimeError('animation/raw renderer object counts disagree')
    # Both native captures traverse the same per-owner list. Reject misalignment.
    for owner in range(2):
        for index, entity in enumerate(render.objects[owner]):
            obj = groups[owner][index]
            if (obj.ownerIndex, obj.listIndex, obj.x, obj.y) != (owner, index, entity.x, entity.y):
                raise RuntimeError('animation/raw renderer object identity mismatch')
    poses, objects = visible_entities(render, visibility, indexed=True)
    result = []
    for seat in range(2):
        values = []
        world = []
        for index in (seat, 1-seat):
            pose, player = poses[index], players[index]
            world.extend((render.players[index].x, render.players[index].y) if pose.visible else (0., 0.))
            values.extend((1., pose.x / 640, pose.y / 480, pose.facing,
                           player.characterId, player.actionId, player.sequenceId,
                           player.subsequenceId, player.animationFrame, player.elapsedInSubsequence)
                          if pose.visible else (0.,) * TOKEN_WIDTH)
        candidates = [(entity.x, entity.y, owner != seat, index, owner, entity)
                      for owner in range(2) for index, entity in objects[owner]]
        # Select by screen position only, never by internal threat/collision state.
        for _, _, _, index, owner, entity in sorted(candidates)[:slots]:
            obj = groups[owner][index]
            native = render.objects[owner][index]
            world.extend((native.x, native.y))
            values.extend((1., entity.x / 640, entity.y / 480, entity.facing,
                           players[owner].characterId, obj.actionId, obj.actionBlockId,
                           obj.animationCounter, obj.animationSubFrame, int(owner != seat)))
        values.extend((0.,) * (TOKEN_WIDTH * slots - (len(values)-20)))
        world.extend((0.,) * (2 * (2+slots) - len(world)))
        values.extend((quantize_gauge(max(0, players[seat].hp), 10000, visibility.hp_quantum),
                       quantize_gauge(max(0, players[1-seat].hp), 10000, visibility.hp_quantum),
                       quantize_spirit(players[seat].spirit, players[seat].maxSpirit, visibility.spirit_quantum),
                       quantize_spirit(players[1-seat].spirit, players[1-seat].maxSpirit, visibility.spirit_quantum),
                       min(1., raw.frameId / horizon)))
        result.append(AnimationObservation(int(raw.frameId), tuple(values), tuple(world)))
    return tuple(result)


def public_history(values, slots, world_coordinates=False):
    """Explicit lossy adapter for existing public rule opponents, not the learner."""
    from soku_rl.env.observation.visible_state import STATE_FEATURES, OBJECT_SLOTS
    frames = np.asarray(values).reshape(-1, frame_width(slots, world_coordinates))[:, :frame_width(slots)]
    output = np.zeros((len(frames), STATE_FEATURES), np.float32)
    for n, frame in enumerate(frames):
        global_values = frame[-5:]
        for seat in range(2):
            token = frame[seat*10:seat*10+10]
            output[n, seat*8:seat*8+8] = (*token[:4], 1., global_values[seat],
                                        global_values[2+seat], token[4]/19)
        tokens = frame[20:-5].reshape(slots, 10)
        for owner in range(2):
            selected = tokens[(tokens[:, 0] > 0) & (tokens[:, 9] == owner)]
            start = 16 + owner * OBJECT_SLOTS * 3
            output[n, start:start+len(selected)*3] = selected[:, :3].reshape(-1)
    return output.reshape(-1)

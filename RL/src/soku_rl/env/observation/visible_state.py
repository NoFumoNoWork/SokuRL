"""Encode public screen positions and quantized gauges without engine action IDs."""
from dataclasses import dataclass

from soku_rl.env.observation.visibility import visible_entities, quantize_gauge, quantize_spirit


OBJECT_SLOTS = 64
STATE_FEATURES = 2 * 8 + 2 * OBJECT_SLOTS * 3


@dataclass(frozen=True, slots=True)
class StateObservation:
    frame: int
    values: tuple[float, ...]

    def __post_init__(self):
        if len(self.values) != STATE_FEATURES:
            raise ValueError("incorrect public state feature count")


def observe_visible_states(raw, render, config):
    poses, objects = visible_entities(render, config)
    try:
        first = _encode(raw, poses, objects, 0, config)
        values = first.values
        second = StateObservation(first.frame,
            values[8:16] + values[:8] + values[16 + OBJECT_SLOTS * 3:] + values[16:16 + OBJECT_SLOTS * 3])
        return first, second
    except ValueError as error:
        fighters = [{key: getattr(fighter, key) for key in
                     ("characterId", "hp", "spirit", "maxSpirit")}
                    for fighter in (raw.p1, raw.p2)]
        raise ValueError(f"public state failed at frame={raw.frameId}, "
                         f"fighters={fighters}: {error}") from error


def _encode(raw, poses, objects, player, config):
    players = (raw.p1, raw.p2)
    values = []
    for index in (player, 1 - player):
        pose, fighter = poses[index], players[index]
        # Health and spirit bars are public; cards and hidden weather IDs are absent.
        hp = quantize_gauge(max(0, fighter.hp), 10000, config.hp_quantum)
        spirit = quantize_spirit(fighter.spirit, fighter.maxSpirit, config.spirit_quantum)
        values.extend((float(pose.visible), pose.x / 640, pose.y / 480, float(pose.facing),
                       1., hp, spirit, fighter.characterId / 19))
    for index in (player, 1 - player):
        # Visibility uses every captured contour before limiting public slots.
        # Keep the existing screen-coordinate ordering and model input shape.
        visible = objects[index][:OBJECT_SLOTS]
        for entity in visible:
            values.extend((1., entity.x / 640, entity.y / 480))
        values.extend([0.] * ((OBJECT_SLOTS - len(visible)) * 3))
    return StateObservation(int(raw.frameId), tuple(values))

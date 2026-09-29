"""Encode public screen positions and quantized gauges without engine action IDs."""
from dataclasses import dataclass

from .render_state import MAX_OBJECTS
from .visibility import visible_entities, quantize_gauge, quantize_spirit


STATE_FEATURES = 2 * 8 + 2 * MAX_OBJECTS * 3


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
        return tuple(_encode(raw, poses, objects, player, config) for player in (0, 1))
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
        visible = objects[index]
        for entity in visible:
            values.extend((1., entity.x / 640, entity.y / 480))
        values.extend([0.] * ((MAX_OBJECTS - len(visible)) * 3))
    return StateObservation(int(raw.frameId), tuple(values))

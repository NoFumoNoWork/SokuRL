"""Project private renderer metadata into quantized screen-space observations.

This module handles viewport, alpha and weather restrictions. Final pixel
occlusion is a separate evidence requirement; drawable is not visibility proof.
"""
from dataclasses import dataclass
from math import floor, isfinite


@dataclass(frozen=True)
class VisibilityConfig:
    pixel_quantum: int
    minimum_alpha: float
    hp_quantum: float
    spirit_quantum: float

    def __post_init__(self):
        if type(self.pixel_quantum) is not int or not 1 <= self.pixel_quantum <= 64:
            raise ValueError("pixel_quantum must be an integer in [1,64]")
        for value in (self.minimum_alpha, self.hp_quantum, self.spirit_quantum):
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError("visibility fractions must be finite and in (0,1]")


@dataclass(frozen=True, slots=True)
class ScreenEntity:
    visible: bool
    x: float
    y: float
    facing: int


HIDDEN_ENTITY = ScreenEntity(False, 0., 0., 0)


def screen_entity(entity, scene, config):
    numbers = (entity.x, entity.y, entity.alpha, scene.camera_x, scene.camera_y, scene.camera_scale)
    if not all(isfinite(value) for value in numbers) or scene.camera_scale <= 0:
        raise ValueError("invalid renderer geometry")
    if not 0 <= entity.alpha <= 1 or entity.drawable not in (0, 1):
        raise ValueError("invalid renderer opacity or drawable flag")
    # Until each weather effect is checked against pixels, it supplies no pose
    # information. CLEAR is enum value 21 in the supported game version.
    if scene.weather != 21 or not entity.drawable or entity.alpha < config.minimum_alpha:
        return HIDDEN_ENTITY
    x = (entity.x + scene.camera_x) * scene.camera_scale
    y = (scene.camera_y - entity.y) * scene.camera_scale
    if not 0 <= x < 640 or not 0 <= y < 480:
        return HIDDEN_ENTITY
    quantum = config.pixel_quantum
    x = min(640 - quantum, floor(x / quantum + 0.5) * quantum)
    y = min(480 - quantum, floor(y / quantum + 0.5) * quantum)
    return ScreenEntity(True, float(x), float(y), entity.facing)


def visible_entities(scene, config):
    if scene.overflow:
        raise RuntimeError("renderer object metadata overflow")
    players = tuple(screen_entity(entity, scene, config) for entity in scene.players)
    objects = []
    for group in scene.objects:
        # Hidden list positions and counts must not survive into public features.
        visible = [screen_entity(entity, scene, config) for entity in group]
        objects.append(tuple(sorted((entity for entity in visible if entity.visible),
                                    key=lambda entity: (entity.x, entity.y, entity.facing))))
    return players, tuple(objects)


def quantize_gauge(value, maximum, quantum):
    if not isfinite(value) or not isfinite(maximum) or maximum <= 0 or not 0 <= value <= maximum:
        raise ValueError("invalid visible gauge value")
    return min(1., max(0., floor(value / maximum / quantum + 0.5) * quantum))

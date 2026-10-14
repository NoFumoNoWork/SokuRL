"""Project private renderer metadata into quantized screen-space observations.

Contours and mutual overlap are approximations, not pixel segmentation.
"""
from dataclasses import dataclass
from math import floor, isfinite
from soku_rl.env.observation.contours import Contour, visible_fraction
from soku_rl.env.observation.gauges import spirit_fraction


@dataclass(frozen=True)
class VisibilityConfig:
    pixel_quantum: int
    minimum_alpha: float
    hp_quantum: float
    spirit_quantum: float
    player_width: float
    player_height: float
    object_diameter: float
    minimum_visible_fraction: float

    def __post_init__(self):
        if type(self.pixel_quantum) is not int or not 1 <= self.pixel_quantum <= 64:
            raise ValueError("pixel_quantum must be an integer in [1,64]")
        for value in (self.minimum_alpha, self.hp_quantum, self.spirit_quantum,
                      self.minimum_visible_fraction):
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError("visibility fractions must be finite and in (0,1]")
        for value in (self.player_width, self.player_height, self.object_diameter):
            if not isfinite(value) or not 0 < value <= 640:
                raise ValueError("contour dimensions must be finite and in (0,640]")


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
    # Weather is not an all-screen invisibility switch. Card visibility is not
    # part of these features; pose visibility follows renderer alpha/geometry.
    if not entity.drawable or entity.alpha < config.minimum_alpha:
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
    raw = (*scene.players, *scene.objects[0], *scene.objects[1])
    projected = tuple(screen_entity(entity, scene, config) for entity in raw)
    contours = []
    for index, (entity, pose) in enumerate(zip(raw, projected)):
        if not pose.visible:
            contours.append(None)
            continue
        scale = scene.camera_scale
        width = config.player_width if index < 2 else config.object_diameter
        height = config.player_height if index < 2 else config.object_diameter
        x = (entity.x + scene.camera_x) * scale
        y = (scene.camera_y - entity.y) * scale
        # Character origins are feet; object origins use a centered disk.
        contours.append(Contour(x, y - height * scale / 2 if index < 2 else y,
                                width * scale / 2, height * scale / 2, entity.alpha))
    filtered = tuple(
        pose if contour is not None and visible_fraction(
            contour, (other for j, other in enumerate(contours)
                      if j != i and other is not None)) >= config.minimum_visible_fraction
        else HIDDEN_ENTITY
        for i, (pose, contour) in enumerate(zip(projected, contours)))
    players = filtered[:2]
    objects = []
    offset = 2
    for group in scene.objects:
        # Hidden list positions and counts must not survive into public features.
        visible = filtered[offset:offset + len(group)]
        offset += len(group)
        objects.append(tuple(sorted((entity for entity in visible if entity.visible),
                                    key=lambda entity: (entity.x, entity.y, entity.facing))))
    return players, tuple(objects)


def quantize_gauge(value, maximum, quantum):
    if not isfinite(value) or not isfinite(maximum) or maximum <= 0 or not 0 <= value <= maximum:
        raise ValueError(f"invalid visible gauge value: value={value}, maximum={maximum}, quantum={quantum}")
    return min(1., max(0., floor(value / maximum / quantum + 0.5) * quantum))


def quantize_spirit(value, maximum, quantum):
    """Project the bridge's signed spirit value into its visible gauge."""
    return quantize_gauge(spirit_fraction(value, maximum), 1, quantum)

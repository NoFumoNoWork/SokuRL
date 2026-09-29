"""Decode renderer metadata; these values are private inputs to visibility filtering."""
from dataclasses import dataclass
import struct

from .pixels import RGBFrame


MAX_OBJECTS = 64
CAMERA = struct.Struct("<fffI")
ENTITY = struct.Struct("<fffiI")
COUNTS = struct.Struct("<III")
RENDER_STATE_SIZE = CAMERA.size + ENTITY.size * (2 + 2 * MAX_OBJECTS) + COUNTS.size


@dataclass(frozen=True, slots=True)
class RenderEntity:
    x: float
    y: float
    alpha: float
    facing: int
    drawable: int


def decode_entities(data, start, count):
    return tuple(RenderEntity(*ENTITY.unpack_from(data, CAMERA.size + i * ENTITY.size))
                 for i in range(start, start + count))


@dataclass(frozen=True, slots=True)
class RenderSnapshot:
    camera_x: float
    camera_y: float
    camera_scale: float
    weather: int
    players: tuple[RenderEntity, RenderEntity]
    objects: tuple[tuple[RenderEntity, ...], tuple[RenderEntity, ...]]
    overflow: bool

    @classmethod
    def decode(cls, data):
        if len(data) != RENDER_STATE_SIZE:
            raise ValueError("invalid render metadata length")
        camera = CAMERA.unpack_from(data)
        count0, count1, overflow = COUNTS.unpack_from(data, len(data) - COUNTS.size)
        if count0 > MAX_OBJECTS or count1 > MAX_OBJECTS or overflow not in (0, 1):
            raise ValueError("invalid render object counts")
        return cls(*camera, decode_entities(data, 0, 2),
                   (decode_entities(data, 2, count0), decode_entities(data, 2 + MAX_OBJECTS, count1)),
                   bool(overflow))


@dataclass(frozen=True, slots=True)
class CapturedScene:
    image: RGBFrame
    render: RenderSnapshot

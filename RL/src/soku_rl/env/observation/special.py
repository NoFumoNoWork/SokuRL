"""Character-specific th123_ai fields, with the source's sentinel semantics."""
import math

from .memory_schema import SPECIAL_FIELDS


def special_values(character, read, previous_objects):
    values = [-1] * 28
    for index, (owner, offset, kind) in SPECIAL_FIELDS.items():
        if owner not in (-1, character):
            continue
        value = int(read(offset, kind))
        if index in (17, 18, 19, 20, 21):
            limit = 3 if index == 21 else 4
            value = value if 0 <= value <= limit else 0
        elif index in (8, 22, 23, 24, 25):
            value = max(0, value)
        elif value == -1:
            value = 0
        values[index] = value
    if character == 10:
        value = read(0x8A0, "i")
        if not 0 <= value <= 3:
            raw = read(0x8A0, "f")
            # x86 conversion of NaN/Inf returns INT_MIN, then the original
            # range check resets it to zero.
            value = int(raw) if math.isfinite(raw) else 0
            if not 0 <= value <= 3:
                value = 0
        values[7] = value
    for index, owner, action, images, lifetime in (
        # Packaged 0.93 uses image 0x1B4 and 601 frames. The later public
        # source changed both constants; preserve the shipped behavior.
        (13, 4, 0x358, (0x154, 0x1B4), 601),
        (14, 4, 0x358, (0x154, 0x1B4), 3001),
        (16, 10, 0x35A, (0xF0, 0x15F), 601)):
        if character == owner:
            obj = next((obj for obj in previous_objects if obj["act"] == action and obj["img"] in images), None)
            values[index] = 0 if obj is None else lifetime + obj["hp"] if index == 14 else lifetime - obj["frame"]
    return tuple(values)

"""Decode exported resource values without changing the native frame format."""
from math import isfinite


def spirit_fraction(value, maximum):
    if type(value) is not int or not -(1 << 15) <= value < (1 << 15):
        raise ValueError(f"invalid signed spirit value: {value}")
    if not isfinite(maximum) or maximum < 0:
        raise ValueError(f"invalid spirit maximum: {maximum}")
    # ABI 8 preserves currentSpirit's signed 16-bit value in an int32 field.
    if value > maximum:
        raise ValueError(f"spirit exceeds maximum: value={value}, maximum={maximum}")
    # All five orbs can be broken. Zero capacity is an empty gauge, not a
    # malformed denominator; keep impossible positive spirit invalid above.
    if maximum == 0:
        return 0.
    return max(0, value) / maximum

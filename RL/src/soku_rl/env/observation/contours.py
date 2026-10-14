"""Estimate visible ellipse area without pixels or engine collision boxes."""
from dataclasses import dataclass


# Fixed quadrature points inside a unit ellipse. No random sampling across frames.
SAMPLES = tuple((x / 4, y / 4) for x in range(-4, 5) for y in range(-4, 5)
                if x * x + y * y < 16)


@dataclass(frozen=True, slots=True)
class Contour:
    x: float
    y: float
    radius_x: float
    radius_y: float
    alpha: float

    def overlaps(self, other):
        return (abs(self.x - other.x) < self.radius_x + other.radius_x
                and abs(self.y - other.y) < self.radius_y + other.radius_y)

    def contains(self, x, y):
        return ((x - self.x) / self.radius_x) ** 2 + ((y - self.y) / self.radius_y) ** 2 < 1


def visible_fraction(target, blockers):
    """Treat overlap as ambiguous: either object may be in front.

    This deliberately removes evidence on both sides of an overlap. The bridge
    does not yet provide draw order. Alpha attenuates covered sample points.
    """
    nearby = tuple(other for other in blockers if target.overlaps(other))
    if (not nearby and 0 <= target.x - target.radius_x and target.x + target.radius_x < 640
            and 0 <= target.y - target.radius_y and target.y + target.radius_y < 480):
        return 1.
    remaining = 0.
    for dx, dy in SAMPLES:
        x, y = target.x + dx * target.radius_x, target.y + dy * target.radius_y
        if not (0 <= x < 640 and 0 <= y < 480):
            continue
        weight = 1.
        for other in nearby:
            if other.contains(x, y):
                weight *= 1 - other.alpha
                if weight == 0.:
                    break
        remaining += weight
    return remaining / len(SAMPLES)

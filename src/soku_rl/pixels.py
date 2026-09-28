"""Immutable RGB frames for transport without a numerical runtime in Wine."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RGBFrame:
    frame: int
    width: int
    height: int
    pixels: bytes

    def __post_init__(self):
        if type(self.frame) is not int or self.frame < 0:
            raise ValueError("image frame must be nonnegative")
        if type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) < 1:
            raise ValueError("image dimensions must be positive integers")
        if type(self.pixels) is not bytes or len(self.pixels) != self.width * self.height * 3:
            raise ValueError("RGB payload length does not match its dimensions")

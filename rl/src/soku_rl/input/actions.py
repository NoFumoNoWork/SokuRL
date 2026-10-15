"""High-level action definitions for the environment."""

from enum import IntEnum


class Action(IntEnum):
    """Initial discrete action vocabulary; combinations will be refined later."""

    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    A = 5
    B = 6
    C = 7
    D = 8

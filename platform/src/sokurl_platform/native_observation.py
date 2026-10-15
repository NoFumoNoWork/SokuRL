"""Native decoded dictionaries, before any policy visibility or feature encoding."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeObservation:
    world: dict
    players: tuple

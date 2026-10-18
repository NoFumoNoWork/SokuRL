"""Learning view over Platform native memory decoding."""
from sokurl_platform.memory import ProcessMemory
from sokurl_platform.native_reader import PrivilegedReader as NativeReader
from soku_rl.env.observation.privileged import PrivilegedObservation

class PrivilegedReader(NativeReader):
    def __init__(self, memory):
        super().__init__(memory, PrivilegedObservation)


class SharedCurrentReader(PrivilegedReader):
    """Shared schema excludes future deck order; never read that private field.

    Zero slots are explicit redaction padding, NOT a measured effective deck.
    The normal complete reader retains its original strict20-card validation.
    """
    def deck(self, seat):
        return (0,)*20

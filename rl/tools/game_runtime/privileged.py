"""Learning view over Platform native memory decoding."""
from sokurl_platform.memory import ProcessMemory
from sokurl_platform.native_reader import PrivilegedReader as NativeReader
from soku_rl.env.observation.privileged import PrivilegedObservation

class PrivilegedReader(NativeReader):
    def __init__(self, memory):
        super().__init__(memory, PrivilegedObservation)

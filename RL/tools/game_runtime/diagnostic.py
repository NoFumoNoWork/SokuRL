"""Complete compact observations when effects fill the bridge's 64 object slots."""
import struct

from soku_rl.env.encoding import MAX_PROJECTILES
from soku_rl.env.observation.diagnostic import Projectile, observe, observe_projectiles
from soku_rl.env.observation.memory_schema import MAX_OBJECTS


class DiagnosticReader:
    def __init__(self, memory):
        self.memory = memory

    def value(self, address, kind):
        values = struct.unpack("<" + kind, self.memory.read(address, struct.calcsize("<" + kind)))
        return values[0] if len(values) == 1 else values

    def projectiles(self, seat):
        root = self.value(0x8985DC, "I")
        head, tail = self.value(root + 0x40, "2I")
        if tail - head != 8:
            raise RuntimeError("expected exactly two character object managers")
        character = self.value(head + seat * 4, "I")
        manager = self.value(character + 0x6F8, "I")
        _, sentinel, count = self.value(manager + 0x58, "3I")
        if count > MAX_OBJECTS:
            raise RuntimeError("complete diagnostic object list exceeds capacity")
        node = self.value(sentinel, "I")
        result, visited = [], set()
        for _ in range(count):
            if node in (0, sentinel) or node in visited:
                raise RuntimeError("invalid diagnostic object list")
            visited.add(node)
            node, _, address = self.value(node, "3I")
            # Pinned SokuLib ProjectileManager::isActive and hitBoxCount;
            # identical filtering to native FrameState.cpp + observe().
            if self.value(address + 0x34C, "i") and self.value(address + 0x1CB, "B"):
                x, y, speed_x, speed_y = self.value(address + 0xEC, "4f")
                result.append(Projectile(x, y, speed_x, speed_y))
        if node != sentinel:
            raise RuntimeError("diagnostic object list exceeded its declared count")
        if len(result) > MAX_PROJECTILES:
            raise RuntimeError("active projectile observation exceeds the declared space")
        return tuple(result)

    def observe(self, raw, bridge):
        if not (raw.p1ObjectOverflow or raw.p2ObjectOverflow):
            return tuple(observe(raw, seat) for seat in (0, 1))
        before = bridge.snapshot()
        if (before.run_state_name != "PAUSED" or before.game_frame != raw.frameId
                or before.latest.segmentId != raw.segmentId):
            raise RuntimeError("diagnostic reads require the exact paused simulation frame")
        self.memory.begin_frame()
        projectiles = tuple(self.projectiles(seat) for seat in (0, 1))
        after = bridge.snapshot()
        if (after.run_state_name != "PAUSED" or after.game_frame != raw.frameId
                or after.latest.segmentId != raw.segmentId):
            raise RuntimeError("game advanced during diagnostic observation")
        return tuple(observe_projectiles(raw, seat, projectiles[1 - seat]) for seat in (0, 1))

    def observe_snapshot(self, raw):
        """Decode immutable realtime capture memory, never the advancing process."""
        if not (raw.p1ObjectOverflow or raw.p2ObjectOverflow):
            return tuple(observe(raw, seat) for seat in (0, 1))
        projectiles = tuple(self.projectiles(seat) for seat in (0, 1))
        return tuple(observe_projectiles(raw, seat, projectiles[1 - seat]) for seat in (0, 1))

    def close(self):
        self.memory.close()

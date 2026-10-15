"""Native object traversal; caller chooses projection type/capacity."""
import struct
from sokurl_platform.native_layout import MAX_OBJECTS

class ProjectileReader:
    def __init__(self, memory, capacity, object_type):
        self.memory, self.capacity, self.object_type = memory, capacity, object_type

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
                result.append(self.object_type(x, y, speed_x, speed_y))
        if node != sentinel:
            raise RuntimeError("diagnostic object list exceeded its declared count")
        if len(result) > self.capacity:
            raise RuntimeError("active projectile observation exceeds the declared space")
        return tuple(result)

"""Read immutable address ranges copied at one original game frame boundary."""
from bisect import bisect_right


class SnapshotMemory:
    def __init__(self, regions, data):
        blocks = {}
        for address, size, offset in regions:
            if (not 0 < address < 2**32 or not 0 < size <= 65536 or
                    address + size > 2**32 or offset < 0 or offset + size > len(data)):
                raise ValueError("invalid captured memory region")
            block = bytes(data[offset:offset + size])
            if address in blocks:
                previous = blocks[address]
                if previous[:min(size, len(previous))] != block[:min(size, len(previous))]:
                    raise ValueError("inconsistent duplicated memory region")
                if len(previous) >= size:
                    continue
            blocks[address] = block
        self.blocks = blocks
        self.addresses = sorted(blocks)

    def begin_frame(self):
        pass

    def read(self, address, size):
        if type(address) is not int or type(size) is not int or not 0 < size <= 65536:
            raise ValueError("invalid snapshot read")
        index = bisect_right(self.addresses, address)
        # Captured fields can lie inside a larger fighter block. Look behind
        # nested short ranges, but never fabricate bytes outside the snapshot.
        for position in range(index - 1, -1, -1):
            start = self.addresses[position]
            block = self.blocks[start]
            offset = address - start
            if offset + size <= len(block):
                return block[offset:offset + size]
            if offset >= 65536:
                break
        raise ValueError(f"uncaptured game memory at {address:#x}, length {size}")

    def close(self):
        pass

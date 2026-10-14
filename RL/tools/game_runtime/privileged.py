"""Read complete th123_ai inputs while an owned offline game is paused."""
import ctypes
from ctypes import wintypes
import struct

from soku_rl.env.observation.memory_schema import FIGHTER_FIELDS, MAX_BOXES, MAX_OBJECTS, OBJECT_FIELDS
from soku_rl.env.observation.privileged import PrivilegedObservation
from soku_rl.env.observation.special import special_values


class ProcessMemory:
    def __init__(self, pid):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.ReadProcessMemory.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.OpenProcess(0x0010, False, pid)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "cannot open owned game for observation")
        self.pages = {}

    def begin_frame(self):
        self.pages.clear()

    def read(self, address, size):
        if type(address) is not int or not 0 < address < 2**32 or not 0 < size <= 65536:
            raise ValueError(f"invalid 32-bit game memory read: address={address!r}, size={size}")
        result = bytearray()
        while size:
            page = address & ~4095
            if page not in self.pages:
                self.pages[page] = self.read_page(page)
            offset = address - page
            count = min(size, 4096 - offset)
            result.extend(self.pages[page][offset:offset + count])
            address += count
            size -= count
        return bytes(result)

    def read_page(self, address):
        size = 4096
        buffer = ctypes.create_string_buffer(size)
        count = ctypes.c_size_t()
        if (not self.kernel.ReadProcessMemory(self.handle, address, buffer, size, ctypes.byref(count))
                or count.value != size):
            raise OSError(ctypes.get_last_error(), f"game memory read failed at {address:#x}")
        return buffer.raw

    def close(self):
        self.kernel.CloseHandle(self.handle)


class PrivilegedReader:
    def __init__(self, memory):
        self.memory = memory
        self.previous = [None, None]
        self.segment = -1

    def value(self, address, kind):
        values = struct.unpack("<" + kind, self.memory.read(address, struct.calcsize("<" + kind)))
        return values[0] if len(values) == 1 else values

    def float_field(self, value, previous):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        exponent = (bits >> 23) & 255
        # th123_ai/lib.cpp leaves persistent fighter fields unchanged when
        # CheckFloat rejects a read, including negative zero and tiny speeds.
        return previous if exponent > 254 or (bits != 0 and exponent < 108) else value

    def entity(self, address, previous):
        data = self.memory.read(address, 0x360)
        read = lambda offset, kind: struct.unpack_from("<" + kind, data, offset)[0]
        entity = {name: read(offset, kind) for name, (offset, kind) in OBJECT_FIELDS.items()}
        for name, (_, kind) in OBJECT_FIELDS.items():
            if kind == "f":
                entity[name] = self.float_field(entity[name], 0. if previous is None else previous[name])
        animation = read(0x150, "I")
        entity.update(img=self.value(animation + 0xA, "I"),
                      fflags=self.value(animation + 0x4C, "I"),
                      aflags=self.value(animation + 0x50, "I"), address=address)
        for key, count_offset, boxes_offset in (("hitarea", 0x1CC, 0x1D0), ("attackarea", 0x1CB, 0x220)):
            count = read(count_offset, "B")
            if count > MAX_BOXES:
                raise RuntimeError("native collision box count exceeds the reference limit")
            boxes = []
            for index in range(count):
                pointer = read(0x320 + index * 4, "I") if key == "attackarea" else 0
                box = list(self.value(pointer if pointer else address + boxes_offset + index * 16, "4i"))
                relative = pointer or (key == "hitarea" and read(0x334 + index * 4, "I"))
                if relative:
                    # Packaged ver0.93 adds coordinates before truncation.
                    # Source ver0.96 truncates first and omits hit-box flags.
                    box[0] = int(box[0] + entity["x"]); box[2] = int(box[2] + entity["x"])
                    box[1] = int(box[1] - entity["y"]); box[3] = int(box[3] - entity["y"])
                box[1] *= -1; box[3] *= -1
                boxes.append(tuple(box))
            entity[key], entity[key + "_n"] = tuple(boxes), count
        return entity

    def objects(self, seat):
        root = self.value(0x8985DC, "I")
        head, tail = self.value(root + 0x40, "2I")
        if tail - head != 8:
            raise RuntimeError("expected exactly two character object managers")
        character = self.value(head + seat * 4, "I")
        manager = self.value(character + 0x6F8, "I")
        _, sentinel, count = self.value(manager + 0x58, "3I")
        if count > MAX_OBJECTS:
            raise RuntimeError("complete object observation exceeds the declared capacity")
        node = self.value(sentinel, "I")
        result = []
        for _ in range(count):
            if node in (0, sentinel):
                raise RuntimeError("object list ended before its declared count")
            node, _, address = self.value(node, "3I")
            result.append(self.entity(address, None))
        if node != sentinel:
            raise RuntimeError("object list exceeded its declared count")
        return tuple(result)

    def deck(self, seat):
        # GameStartParams -> PlayerInfo -> Dequeue<unsigned short>.
        address = 0x899D10 + seat * 0x20 + 8
        _, table, chunks, counter, count = self.value(address, "5I")
        if count != 20 or chunks == 0:
            raise RuntimeError("the selected effective deck must contain twenty cards")
        pointers, cards = {}, []
        for i in range(20):
            position = counter + i
            chunk = (position >> 3) % chunks
            # One paused frame: reuse only this call's repeated table entries.
            # Card values and pointers are read again on the next observation.
            if chunk not in pointers:
                pointers[chunk] = self.value(table + chunk * 4, "I")
            cards.append(self.value(pointers[chunk] + (position & 7) * 2, "H"))
        return tuple(cards)

    def fighter(self, address, seat, weather):
        previous = self.previous[seat]
        entity = self.entity(address, previous)
        data = self.memory.read(address + 0x400, 0x530)
        read = lambda offset, kind: struct.unpack_from("<" + kind, data, offset - 0x400)[0]
        entity.update({name: read(offset, kind) for name, (offset, kind) in FIGHTER_FIELDS.items()})
        for name, (_, kind) in FIGHTER_FIELDS.items():
            if kind == "f":
                entity[name] = self.float_field(entity[name], 0. if previous is None else previous[name])
        entity["char"] = self.value(0x899D10 + seat * 0x20, "I")
        entity["spell"] = 0 if weather == 11 else read(0x5E4, "H") + read(0x5E6, "B") * 500
        entity["card"] = -1 if weather == 11 else 0 if previous is None else previous["cards"][0]
        count, point, table, maximum = read(0x5F8, "i"), read(0x5F4, "I"), read(0x5EC, "I"), read(0x5F0, "I")
        if not 0 <= count <= 5:
            raise RuntimeError("invalid hand card count")
        cards = []
        for index in range(5):
            if weather != 11 and maximum > 0 and index < entity["spell"] // 500:
                card = self.value(table + ((point + index) % maximum) * 4, "I")
                # The original reader leaves this persistent slot unchanged
                # when a card is removed and its table entry becomes null.
                prior = (0, 0) if previous is None else previous["cards"][index * 2:index * 2 + 2]
                cards.extend(prior if card == 0 else self.value(card, "2h"))
            else:
                cards.extend((-1, -1))
        entity["cards"] = tuple(cards)
        entity["skills"] = struct.unpack_from("<16b", data, 0x6C4 - 0x400)
        entity["special"] = special_values(entity["char"], read, () if previous is None else previous["objects"])
        key_manager = self.value(read(0x750, "I"), "I")
        horizontal, vertical, *buttons = self.value(key_manager + 0x38, "8i")
        entity["keys"] = (max(0, -vertical), max(0, vertical), max(0, -horizontal), max(0, horizontal), *buttons)
        entity["objects"] = self.objects(seat)
        entity["obj_n"] = len(entity["objects"])
        entity["deck"] = self.deck(seat)
        entity["is_card_use"] = 0
        if weather != 11 and count > 0:
            # is_CardUse reads the selected table entry directly; unlike the
            # five-card snapshot, it does not wrap the index modulo maximum.
            selected = self.value(table + point * 4, "I")
            cost = self.value(selected + 2, "h")
            cost -= int(weather == 2 and cost > 1)
            entity["is_card_use"] = int(1 <= cost <= entity["spell"] // 500)
        return entity

    def observe(self, raw, client):
        before = client.snapshot()
        if before.run_state_name != "PAUSED" or before.game_frame != raw.frameId:
            raise RuntimeError("privileged reads require the exact paused simulation frame")
        observations = self._decode(raw)
        after = client.snapshot()
        if (after.run_state_name != "PAUSED" or after.game_frame != raw.frameId
                or after.latest.segmentId != raw.segmentId):
            raise RuntimeError("game advanced during privileged observation")
        return observations

    def observe_snapshot(self, raw):
        from game_runtime.snapshot_memory import SnapshotMemory
        if not isinstance(self.memory, SnapshotMemory):
            raise TypeError("realtime observations require immutable captured memory")
        return self._decode(raw)

    def _decode(self, raw):
        self.memory.begin_frame()
        if self.segment != raw.segmentId:
            self.previous = [None, None]
            self.segment = raw.segmentId
        world = {"frame": int(raw.frameId), "battle_time": self.value(0x8985D8, "i"),
                 "weather": self.value(0x8971C0, "i"), "weather2": self.value(0x8971C4, "i"),
                 "weather_time": self.value(0x8971CC, "h"), "stage_number": self.value(0x8971CE, "b"),
                 "bgm_number": self.value(0x899D0D, "b")}
        battle = self.value(0x8985E4, "I")
        players = tuple(self.fighter(self.value(battle + 0xC + seat * 4, "I"), seat, world["weather"])
                        for seat in (0, 1))
        self.previous = players
        return (PrivilegedObservation(world, players), PrivilegedObservation(world, players[::-1]))

    def close(self):
        self.memory.close()

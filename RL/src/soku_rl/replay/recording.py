"""Extract the original game's current replay record at a paused frame."""
import struct

from soku_rl.replay.format import Replay, ReplayMatch, ReplayPlayer


class OriginalReplayReader:
    def __init__(self, memory):
        self.memory = memory

    def values(self, address, kind):
        return struct.unpack("<" + kind, self.memory.read(address, struct.calcsize("<" + kind)))

    def words(self, address, maximum):
        _, table, chunks, start, count = self.values(address, "5I")
        if count > maximum or (count and (not table or not chunks or start >= chunks * 8)):
            raise RuntimeError("invalid original replay deque")
        result = []
        for index in range(count):
            position = start + index
            pointer, = self.values(table + ((position // 8) % chunks) * 4, "I")
            result.extend(self.values(pointer + (position % 8) * 2, "H"))
        return tuple(result)

    def read(self, client):
        before = client.snapshot()
        if before.run_state_name != "PAUSED" or before.latest.battleSubMode != 1:
            raise RuntimeError("replay extraction requires a paused game in recording mode")
        self.memory.begin_frame()
        manager = 0x898718
        record, = self.values(manager + 0x104, "I")
        if not record:
            raise RuntimeError("the original game did not create a replay record")
        header = bytearray(self.memory.read(manager + 0xE8, 10))
        header[7] = 1  # Export the current episode, not previous matches.
        # The normal save path fills this field after recording. It selects
        # which input streams playback consumes; zero would enable CPU control.
        input_managers = self.values(manager, "2I")
        header[8] = sum(int(pointer != 0) << seat for seat, pointer in enumerate(input_managers))
        if header[8] != 3:
            raise RuntimeError("environment replays require two recorded input streams")
        players = []
        for seat in (0, 1):
            character, = self.values(record + seat, "B")
            palette, = self.values(record + 2 + seat, "B")
            extra, = self.values(record + 4 + seat, "B")
            side, = self.values(record + 6 + seat, "B")
            input_type, = self.values(record + 0x30 + seat, "B")
            cards = self.words(record + 8 + seat * 20, 20)
            players.append(ReplayPlayer(character, palette, cards, side, extra, input_type))
        mode, stage, music = self.values(record + 0x32, "3B")
        seed, = self.values(record + 0x38, "I")
        match = ReplayMatch(tuple(players), mode, stage, music, seed,
                            self.words(record + 0x3C, 20_000_000))
        enabled, = self.values(0x88291C, "B")
        after = client.snapshot()
        if (after.run_state_name != "PAUSED" or after.game_frame != before.game_frame
                or after.latest.segmentId != before.latest.segmentId):
            raise RuntimeError("the game advanced during replay extraction")
        return Replay(0x100D2 if enabled else 0xD2, bytes(header), (match,))

"""Read and write the original th123 1.10a replay byte layout."""
from dataclasses import dataclass
import struct


@dataclass(frozen=True)
class ReplayPlayer:
    character: int
    palette: int
    cards: tuple
    side: int
    info_byte6: int
    input_type: int

    def __post_init__(self):
        if not 0 <= self.character < 20 or not 0 <= self.palette < 8:
            raise ValueError("replay requires original playable characters and palettes")
        if len(self.cards) > 20 or any(type(v) is not int or not 0 <= v <= 65535 for v in self.cards):
            raise ValueError("invalid replay deck")
        if self.side not in (0, 1) or any(not 0 <= v <= 255 for v in (self.info_byte6, self.input_type)):
            raise ValueError("invalid replay player metadata")


@dataclass(frozen=True)
class ReplayMatch:
    players: tuple
    mode: int
    stage: int
    music: int
    seed: int
    input_words: tuple

    def __post_init__(self):
        if len(self.players) != 2 or not all(isinstance(p, ReplayPlayer) for p in self.players):
            raise ValueError("a replay match requires both players")
        if any(type(v) is not int or not 0 <= v <= 255 for v in (self.mode, self.stage, self.music)):
            raise ValueError("invalid replay match metadata")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("replay seed must be a uint32")
        if any(type(v) is not int or not 0 <= v <= 65535 for v in self.input_words):
            raise ValueError("replay input records must be uint16 values")


@dataclass(frozen=True)
class Replay:
    version: int
    header: bytes
    matches: tuple

    def __post_init__(self):
        if self.version not in (0xD2, 0x100D2) or len(self.header) != 10:
            raise ValueError("unsupported th123 replay header")
        if not 1 <= len(self.matches) <= 127 or self.header[7] != len(self.matches):
            raise ValueError("replay match count differs from its header")

    def encode(self):
        parts = [struct.pack("<I", self.version), self.header]
        for match in self.matches:
            for player in match.players:
                parts.append(struct.pack("<BBI", player.character, player.palette, len(player.cards)))
                parts.append(struct.pack("<" + "H" * len(player.cards), *player.cards))
                parts.append(struct.pack("<3B", player.side, player.info_byte6, player.input_type))
            parts.append(struct.pack("<3BII", match.mode, match.stage, match.music, match.seed, len(match.input_words)))
            parts.append(struct.pack("<" + "H" * len(match.input_words), *match.input_words))
        return b"".join(parts)

    @classmethod
    def decode(cls, data):
        cursor = 0

        def read(kind):
            nonlocal cursor
            size = struct.calcsize("<" + kind)
            if cursor + size > len(data):
                raise ValueError("truncated replay record")
            values = struct.unpack_from("<" + kind, data, cursor)
            cursor += size
            return values

        version, = read("I")
        header, = read("10s")
        if version not in (0xD2, 0x100D2) or not 1 <= header[7] <= 127:
            raise ValueError("unsupported th123 replay header")
        matches = []
        for _ in range(header[7]):
            players = []
            for _ in range(2):
                character, palette, count = read("BBI")
                if count > 20:
                    raise ValueError("replay deck exceeds twenty cards")
                cards = read("H" * count)
                players.append(ReplayPlayer(character, palette, cards, *read("3B")))
            mode, stage, music, seed, count = read("3BII")
            if count > (len(data) - cursor) // 2:
                raise ValueError("truncated replay input stream")
            matches.append(ReplayMatch(tuple(players), mode, stage, music, seed, read("H" * count)))
        if cursor != len(data):
            raise ValueError("unrecognized data after the replay matches")
        return cls(version, header, tuple(matches))

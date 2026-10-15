"""Lossless th123 1.10a replay input export, without running the game."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "logs/replay_inputs"
FIELDS = ("horizontal", "vertical", "A", "B", "C", "D", "changeCard", "spellcard")


class Reader:
    def __init__(self, data: bytes):
        self.data, self.offset = data, 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.offset + size > len(self.data):
            raise ValueError(f"truncated replay at byte {self.offset}, requesting {size}")
        result = self.data[self.offset:self.offset + size]
        self.offset += size
        return result

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]


def decode_keys(word: int) -> dict[str, int]:
    # 0x40A370 gives LEFT/UP priority for simultaneous opposing directions.
    return dict(zip(FIELDS, (
        -1 if word & 4 else 1 if word & 8 else 0,
        -1 if word & 1 else 1 if word & 2 else 0,
        *(int(bool(word & (1 << bit))) for bit in range(4, 10)),
    )))


def decode(data: bytes) -> dict:
    reader = Reader(data)
    version = reader.u32()
    if version != 0xD2:
        raise ValueError(f"unsupported replay version 0x{version:X}; expected 1.10a 0xD2")
    header = reader.take(10)
    rounds = []
    for index in range(header[7]):
        start = reader.offset
        players = []
        for _ in range(2):
            character, palette = reader.u8(), reader.u8()
            count = reader.u32()
            cards = list(struct.unpack(f"<{count}H", reader.take(count * 2)))
            options = reader.take(3)
            players.append({"character": character, "palette": palette,
                            "cards": cards, "options_hex": options.hex()})
        stage, music, unknown = reader.u8(), reader.u8(), reader.u8()
        seed, count = reader.u32(), reader.u32()
        words = list(struct.unpack(f"<{count}H", reader.take(count * 2)))
        rounds.append({"round_index": index, "file_offset": start,
                       "players": players, "stage": stage, "music": music,
                       "round_byte_34": unknown, "seed": seed, "input_words": words})
    if reader.offset != len(data):
        raise ValueError(f"unexpected trailing bytes at {reader.offset}/{len(data)}")
    return {"version": version, "header_hex": header.hex(),
            "input_player_mask": header[8], "rounds": rounds}


def encode(replay: dict) -> bytes:
    chunks = [struct.pack("<I", replay["version"]), bytes.fromhex(replay["header_hex"])]
    for round_data in replay["rounds"]:
        for player in round_data["players"]:
            cards = player["cards"]
            chunks.extend((bytes((player["character"], player["palette"])),
                           struct.pack("<I", len(cards)), struct.pack(f"<{len(cards)}H", *cards),
                           bytes.fromhex(player["options_hex"])))
        words = round_data["input_words"]
        chunks.extend((bytes((round_data["stage"], round_data["music"], round_data["round_byte_34"])),
                       struct.pack("<II", round_data["seed"], len(words)),
                       struct.pack(f"<{len(words)}H", *words)))
    return b"".join(chunks)


def poll_rows(replay: dict):
    sides = [side for side in range(2) if replay["input_player_mask"] & (1 << side)]
    if not sides:
        raise ValueError("replay has no input-bearing player")
    for round_data in replay["rounds"]:
        words = round_data["input_words"]
        for start in range(0, len(words), len(sides)):
            row = {"round_index": round_data["round_index"], "input_poll_index": start // len(sides),
                   "complete_poll": int(start + len(sides) <= len(words))}
            for offset, side in enumerate(sides):
                if start + offset >= len(words):
                    continue
                word = words[start + offset]
                prefix = f"p{side + 1}_"
                row[prefix + "word"] = word
                row[prefix + "non_gameplay_bits"] = word & ~0x3FF
                row.update({prefix + k: v for k, v in decode_keys(word).items()})
            yield row


def export(path: Path, output: Path) -> dict:
    data = path.read_bytes()
    replay = decode(data)
    rebuilt = encode(replay)
    if rebuilt != data:
        raise ValueError("replay byte roundtrip failed")
    destination = output / path.stem
    destination.mkdir(parents=True, exist_ok=True)
    replay.update({"source": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                   "byte_roundtrip_verified": True, "live_input_injection_verified": False,
                   "index_semantics": "native input poll order, NOT Bridge battle frameId",
                   "conditions": {"weather": "Clear", "cards_disabled": True,
                                  "origin": "user-required capture conditions; not encoded by this export"}})
    report_path = destination / "matched_conditions.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("replay_sha256") == replay["sha256"]:
            replay["native_playback_validation"] = {
                "report": str(report_path.resolve()), "frames_compared": report["frames_compared"],
                "battle_projection_parity_verified": bool(report["frames_compared"] and
                    report["first_battle_projection_difference"] is None and report["dropped_frames"] == 0),
                "boundary": report["boundary"],
            }
    (destination / "inputs.json").write_text(json.dumps(replay, indent=2), encoding="utf-8")
    # Rebuild from the serialized artifact, not the in-memory decoder output.
    serialized = json.loads((destination / "inputs.json").read_text(encoding="utf-8"))
    if encode(serialized) != data:
        raise ValueError("serialized JSON byte roundtrip failed")
    (destination / "reconstructed.rep").write_bytes(encode(serialized))
    fields = ["round_index", "input_poll_index", "complete_poll"]
    fields += [f"p{side}_{name}" for side in (1, 2)
               for name in ("word", "non_gameplay_bits", *FIELDS)]
    rows = list(poll_rows(replay))
    with (destination / "inputs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    csv_rebuilt = rebuild_from_csv(serialized, destination / "inputs.csv")
    if csv_rebuilt != data:
        raise ValueError("CSV input word roundtrip failed")
    return {"replay": path.name, "sha256": replay["sha256"], "rounds": len(replay["rounds"]),
            "input_words": sum(len(r["input_words"]) for r in replay["rounds"]),
            "complete_input_polls": sum(r["complete_poll"] for r in rows),
            "incomplete_input_polls": sum(not r["complete_poll"] for r in rows),
            "byte_roundtrip_verified": True, "live_input_injection_verified": False}


def rebuild_from_csv(metadata: dict, path: Path) -> bytes:
    replay = deepcopy(metadata)
    for round_data in replay["rounds"]:
        round_data["input_words"] = []
    sides = [side for side in range(2) if replay["input_player_mask"] & (1 << side)]
    expected_poll = [0] * len(replay["rounds"])
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            index = int(row["round_index"])
            if not 0 <= index < len(replay["rounds"]):
                raise ValueError("invalid CSV round index")
            if int(row["input_poll_index"]) != expected_poll[index]:
                raise ValueError("non-contiguous CSV input polls")
            expected_poll[index] += 1
            for side in sides:
                value = row[f"p{side + 1}_word"]
                if value != "":
                    replay["rounds"][index]["input_words"].append(int(value))
    return encode(replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rebuild-json", type=Path)
    parser.add_argument("--csv", type=Path, help="Rebuild using CSV raw input words instead of JSON words")
    parser.add_argument("--replay-output", type=Path)
    args = parser.parse_args()
    if args.rebuild_json:
        if not args.replay_output:
            parser.error("--rebuild-json requires --replay-output")
        metadata = json.loads(args.rebuild_json.read_text(encoding="utf-8"))
        data = rebuild_from_csv(metadata, args.csv) if args.csv else encode(metadata)
        args.replay_output.parent.mkdir(parents=True, exist_ok=True)
        if args.replay_output.exists():
            raise FileExistsError("refusing to overwrite an existing replay")
        args.replay_output.write_bytes(data)
        print(json.dumps({"output": str(args.replay_output.resolve()), "sha256": hashlib.sha256(data).hexdigest()}))
        return
    from sokurl_platform.resources import game_directory
    paths = args.replays or sorted((game_directory() / 'replay').glob("261008_*_reimu_marisa.rep"))
    if not paths:
        raise ValueError("no replay files found")
    result = [export(path, args.output) for path in paths]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

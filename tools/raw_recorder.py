from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bridge_shared import LogicalInput, PlayerState, RawFrameState

INPUT_NAMES = ("horizontalAxis", "verticalAxis", "a", "b", "c", "d", "changeCard", "spellcard")
PLAYER_SCALARS = (
    "characterId", "x", "y", "speedX", "speedY", "facing", "hp", "spirit",
    "maxSpirit", "cardGauge", "cardCount", "actionId", "sequenceId", "subsequenceId",
    "animationFrame", "elapsedInSubsequence", "hitstop", "untech", "airborne",
    "frameFlags", "attackFlags", "objectCount",
)
GLOBAL_FIELDS = (
    "frameId", "segmentId", "sceneId", "battleMode", "battleSubMode", "stageId",
    "roundId", "timeElapsedRaw", "activeWeather", "displayedWeather", "weatherCounter",
    "randomSeed", "stateHash",
)
FRAME_FIELDS = list(GLOBAL_FIELDS)
for _prefix in ("p1", "p2"):
    FRAME_FIELDS.extend(f"{_prefix}_{name}" for name in PLAYER_SCALARS)
    FRAME_FIELDS.extend(f"{_prefix}_hand_{index}" for index in range(5))

INPUT_FIELDS = ["frameId", "segmentId"]
for _prefix in ("p1", "p2"):
    INPUT_FIELDS.extend(f"{_prefix}_{name}" for name in INPUT_NAMES)


def _player_values(player: PlayerState, prefix: str) -> dict[str, int | float]:
    values = {f"{prefix}_{name}": getattr(player, name) for name in PLAYER_SCALARS}
    values.update({f"{prefix}_hand_{index}": player.handIds[index] for index in range(5)})
    return values


def frame_row(state: RawFrameState) -> dict[str, int | float]:
    row = {name: getattr(state, name) for name in GLOBAL_FIELDS}
    row.update(_player_values(state.p1, "p1"))
    row.update(_player_values(state.p2, "p2"))
    return row


def _input_values(value: LogicalInput, prefix: str) -> dict[str, int]:
    return {f"{prefix}_{name}": getattr(value, name) for name in INPUT_NAMES}


def input_row(state: RawFrameState) -> dict[str, int]:
    row = {"frameId": state.frameId, "segmentId": state.segmentId}
    row.update(_input_values(state.p1.input, "p1"))
    row.update(_input_values(state.p2.input, "p2"))
    return row


class RawSessionWriter:
    def __init__(self, root: Path, metadata: dict[str, object] | None = None) -> None:
        now = datetime.now(timezone.utc)
        self.session_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = root / self.session_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.frame_count = 0
        self.first_frame: int | None = None
        self.last_frame: int | None = None
        self._frames_file = (self.path / "frames_000.csv").open("w", newline="", encoding="utf-8")
        self._inputs_file = (self.path / "inputs_000.csv").open("w", newline="", encoding="utf-8")
        self._frames = csv.DictWriter(self._frames_file, fieldnames=FRAME_FIELDS)
        self._inputs = csv.DictWriter(self._inputs_file, fieldnames=INPUT_FIELDS)
        self._frames.writeheader()
        self._inputs.writeheader()
        self._started = now
        payload = {
            "schema_version": 1,
            "session_id": self.session_id,
            "started_utc": now.isoformat(),
            "frame_semantics": "checkpoint-relative BattleManager simulation updates",
            "hash": "FNV-1a-64 over documented deterministic RawFrameState subset",
            **(metadata or {}),
        }
        (self.path / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write(self, frames: Iterable[RawFrameState]) -> int:
        count = 0
        for state in frames:
            self._frames.writerow(frame_row(state))
            self._inputs.writerow(input_row(state))
            frame_id = int(state.frameId)
            self.first_frame = frame_id if self.first_frame is None else self.first_frame
            self.last_frame = frame_id
            self.frame_count += 1
            count += 1
        if count:
            self._frames_file.flush()
            self._inputs_file.flush()
        return count

    def close(self, *, dropped_frames: int, validation: str) -> None:
        if self._frames_file.closed:
            return
        self._frames_file.close()
        self._inputs_file.close()
        ended = datetime.now(timezone.utc)
        manifest = {
            "schema_version": 1,
            "session_id": self.session_id,
            "started_utc": self._started.isoformat(),
            "ended_utc": ended.isoformat(),
            "frames": self.frame_count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "dropped_frames": dropped_frames,
            "valid_lossless_recording": dropped_frames == 0,
            "deterministic_validation": validation,
            "files": ["metadata.json", "frames_000.csv", "inputs_000.csv", "manifest.json"],
        }
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

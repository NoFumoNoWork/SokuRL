from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "ken_oki_microgame_v0_4b_verified_spacing.json"
)


@dataclass(frozen=True)
class FrameData:
    """Typed wrapper around the Ken43 JSON frame-data document."""

    raw: dict[str, Any]
    path: Path

    @property
    def moves(self) -> dict[str, dict[str, Any]]:
        return self.raw["moves"]

    @property
    def system_rules(self) -> dict[str, Any]:
        return self.raw["system_rules"]

    @property
    def movement(self) -> dict[str, Any]:
        return self.raw["movement"]

    @property
    def jump(self) -> dict[str, Any]:
        return self.raw["jump"]

    @property
    def distance_model(self) -> dict[str, Any]:
        return self.raw["distance_model"]

    def move(self, move_id: str) -> dict[str, Any]:
        try:
            return self.moves[move_id]
        except KeyError as exc:
            raise KeyError(f"unknown move: {move_id}") from exc


def load_frame_data(path: str | Path | None = None) -> FrameData:
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    with data_path.open("r", encoding="utf-8") as f:
        return FrameData(raw=json.load(f), path=data_path)

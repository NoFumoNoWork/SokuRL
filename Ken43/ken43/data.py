from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "ken_oki_microgame_v0_4b_verified_spacing.json"
)
DEFAULT_COMBAT_EFFECTS_PATH = (
    Path(__file__).resolve().parents[1] / "ken43_combat_effects_v0.json"
)


@dataclass(frozen=True)
class FrameData:
    """Typed wrapper around the Ken43 JSON frame-data document."""

    raw: dict[str, Any]
    path: Path
    combat_effects: dict[str, Any]

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

    def effects(self, move_id: str) -> dict[str, Any]:
        return self.combat_effects.get("moves", {}).get(move_id, {})

    @property
    def combat_system(self) -> dict[str, Any]:
        return self.combat_effects.get("system", {})

    def validate_combat_effects(self) -> None:
        effects_by_move = self.combat_effects.get("moves", {})
        for move_id, effects in effects_by_move.items():
            if move_id not in self.moves:
                raise ValueError(f"combat effects reference unknown move: {move_id}")
            drive = effects.get("drive")
            scaling = effects.get("scaling")
            if not isinstance(drive, list) or len(drive) != 4:
                raise ValueError(f"{move_id} drive effects must have four fields")
            if not isinstance(scaling, list) or len(scaling) != 3:
                raise ValueError(f"{move_id} scaling effects must have three fields")
            for value in [*drive, *scaling]:
                if value is not None and float(value) < 0:
                    raise ValueError(f"{move_id} combat effects cannot be negative")


def load_frame_data(path: str | Path | None = None) -> FrameData:
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    with data_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    with DEFAULT_COMBAT_EFFECTS_PATH.open("r", encoding="utf-8") as f:
        combat_effects = json.load(f)
    result = FrameData(raw=raw, path=data_path, combat_effects=combat_effects)
    result.validate_combat_effects()
    return result

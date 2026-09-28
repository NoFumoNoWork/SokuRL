from __future__ import annotations

import argparse
import ctypes
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bridge_shared import (
    ACTION_INPUTS,
    BridgeClient,
    BridgeUnavailable,
    LogicalInput,
    ReconstructionFrame,
    SimplePlayerState,
    SimpleStatePatch,
)
from frame_validation import InputPair, PracticeInstance, launch_checkpoint
import sokurl


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = ROOT / "anchors"
ANCHOR_FORMAT = "SokuRLAnchor/v1"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
INPUT_NAMES = (
    "horizontalAxis", "verticalAxis", "a", "b", "c", "d", "changeCard", "spellcard",
)
PLAYER_SIMPLE_NAMES = (
    "x", "y", "speedX", "speedY", "facing", "hp", "spirit", "maxSpirit",
    "cardGauge", "cardCount",
)
NEUTRAL = ACTION_INPUTS["NEUTRAL"]


def anchor_path(name: str) -> Path:
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError("anchor name may contain only letters, digits, '_' and '-'")
    return ANCHOR_DIR / f"{name}.json"


def logical_to_list(value: LogicalInput) -> list[int]:
    return [int(getattr(value, name)) for name in INPUT_NAMES]


def logical_from_list(values: list[int]) -> LogicalInput:
    if len(values) != 8:
        raise ValueError("anchor logical input must have eight fields")
    result = LogicalInput()
    for name, value in zip(INPUT_NAMES, values, strict=True):
        setattr(result, name, value)
    return result


def simple_player_to_dict(value: SimplePlayerState) -> dict[str, int | float]:
    return {name: getattr(value, name) for name in PLAYER_SIMPLE_NAMES}


def simple_patch_to_dict(value: SimpleStatePatch) -> dict[str, object]:
    return {
        "timeElapsedRaw": value.timeElapsedRaw,
        "activeWeather": value.activeWeather,
        "displayedWeather": value.displayedWeather,
        "weatherCounter": value.weatherCounter,
        "p1": simple_player_to_dict(value.p1),
        "p2": simple_player_to_dict(value.p2),
    }


def simple_patch_from_dict(data: dict[str, Any]) -> SimpleStatePatch:
    result = SimpleStatePatch()
    for name in ("timeElapsedRaw", "activeWeather", "displayedWeather", "weatherCounter"):
        setattr(result, name, data[name])
    for player in ("p1", "p2"):
        target = getattr(result, player)
        source = data[player]
        for name in PLAYER_SIMPLE_NAMES:
            setattr(target, name, source[name])
    return result


def reconstruction_to_dict(value: ReconstructionFrame) -> dict[str, object]:
    return {
        "p1_input": logical_to_list(value.p1Input),
        "p2_input": logical_to_list(value.p2Input),
        "simple": simple_patch_to_dict(value.simple),
        "state_hash": f"{value.stateHash:016X}",
    }


def copy_patch(value: SimpleStatePatch) -> SimpleStatePatch:
    result = SimpleStatePatch()
    ctypes.memmove(ctypes.addressof(result), ctypes.addressof(value), ctypes.sizeof(result))
    return result


def save_anchor(name: str, pid: int) -> dict[str, object]:
    with BridgeClient(pid) as client:
        sequence = client.pause()
        snapshot = client.wait_for_ack(sequence)
        if snapshot.ack_seq != sequence or snapshot.run_state_name != "PAUSED":
            raise RuntimeError("could not pause the anchor process")
        if not snapshot.checkpoint_valid:
            raise RuntimeError("process has no valid deterministic checkpoint")
        target = snapshot.game_frame
        history = client.reconstruction_history()
        if target >= len(history):
            raise RuntimeError(
                f"frame {target} is not available in reconstruction history ({len(history)} frames)"
            )
        if history[target].stateHash != snapshot.latest.stateHash:
            raise RuntimeError("live target hash does not match reconstruction history")

        identity = {
            "battle_mode": snapshot.latest.battleMode,
            "battle_submode": snapshot.latest.battleSubMode,
            "stage": snapshot.latest.stageId,
            "random_seed": snapshot.latest.randomSeed,
            "p1_character": snapshot.latest.p1.characterId,
            "p2_character": snapshot.latest.p2.characterId,
            "practice_dummy_state": "2P_CONTROL",
            "checkpoint_hash": f"{history[0].stateHash:016X}",
        }
        document: dict[str, object] = {
            "format": ANCHOR_FORMAT,
            "name": name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": identity,
            "target_frame": target,
            "target_hash": f"{snapshot.latest.stateHash:016X}",
            "setup": [reconstruction_to_dict(frame) for frame in history[:target + 1]],
        }
    path = anchor_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=True), encoding="ascii")
    return document


def read_anchor(name: str) -> dict[str, Any]:
    path = anchor_path(name)
    if not path.is_file():
        raise RuntimeError(f"anchor does not exist: {path}")
    document = json.loads(path.read_text(encoding="ascii"))
    if document.get("format") != ANCHOR_FORMAT:
        raise RuntimeError(f"unsupported anchor format in {path}")
    if document.get("name") != name:
        raise RuntimeError(f"anchor name mismatch in {path}")
    target = document.get("target_frame")
    setup = document.get("setup")
    if not isinstance(target, int) or target < 0 or not isinstance(setup, list) or len(setup) != target + 1:
        raise RuntimeError(f"invalid setup history in {path}")
    return document


def apply_patch(instance: PracticeInstance, patch: SimpleStatePatch) -> int:
    sequence = instance.client.apply_simple_patch(patch)
    snapshot = instance.client.wait_for_ack(sequence)
    if snapshot.ack_seq != sequence or snapshot.result_name != "COMPLETE":
        raise RuntimeError(
            f"simple-state restoration failed at frame {snapshot.game_frame}: {snapshot.result_name}"
        )
    return snapshot.latest.stateHash


def load_anchor(name: str) -> PracticeInstance:
    document = read_anchor(name)
    checkpoint = document["checkpoint"]
    instance = launch_checkpoint(checkpoint["random_seed"])
    try:
        initial = instance.client.snapshot().latest
        actual_identity = (
            initial.battleMode,
            initial.battleSubMode,
            initial.stageId,
            initial.randomSeed,
            initial.p1.characterId,
            initial.p2.characterId,
        )
        expected_identity = (
            checkpoint["battle_mode"],
            checkpoint["battle_submode"],
            checkpoint["stage"],
            checkpoint["random_seed"],
            checkpoint["p1_character"],
            checkpoint["p2_character"],
        )
        if checkpoint.get("practice_dummy_state") != "2P_CONTROL":
            raise RuntimeError("anchor was not created with controllable Practice P2 state")
        if actual_identity != expected_identity:
            raise RuntimeError(
                f"checkpoint identity mismatch: expected={expected_identity} actual={actual_identity}"
            )

        for frame, saved in enumerate(document["setup"]):
            if frame:
                pair = InputPair(tuple(saved["p1_input"]), tuple(saved["p2_input"]))
                instance.step(pair)
            actual_hash = apply_patch(instance, simple_patch_from_dict(saved["simple"]))
            expected_hash = int(saved["state_hash"], 16)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"anchor '{name}' diverged at frame {frame}: "
                    f"expected={expected_hash:016X} actual={actual_hash:016X}"
                )
        final = instance.client.snapshot()
        expected_target = int(document["target_hash"], 16)
        if final.game_frame != document["target_frame"] or final.latest.stateHash != expected_target:
            raise RuntimeError(
                f"anchor target validation failed: frame={final.game_frame} "
                f"hash={final.latest.stateHash:016X}"
            )
        return instance
    except Exception:
        instance.close()
        raise


@dataclass(frozen=True)
class ScenarioScript:
    anchor: str
    opponent: int
    steps: list[dict[str, Any]]


def parse_script(path: Path) -> ScenarioScript:
    anchor = None
    opponent = None
    steps: list[dict[str, Any]] = []
    in_script = False
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not in_script:
            if line == "script:":
                in_script = True
                continue
            if ":" not in line:
                raise ValueError(f"{path}:{number}: expected key: value")
            key, value = (part.strip() for part in line.split(":", 1))
            if key == "anchor":
                anchor = value
            elif key == "opponent":
                opponent = {"p1": 0, "p2": 1}.get(value.casefold())
                if opponent is None:
                    raise ValueError(f"{path}:{number}: opponent must be p1 or p2")
            else:
                raise ValueError(f"{path}:{number}: unknown key {key}")
            continue
        if not line.startswith("-"):
            raise ValueError(f"{path}:{number}: script entries must start with '-'")
        body = line[1:].strip()
        if ":" not in body:
            steps.append({"input": body})
            continue
        key, value = (part.strip() for part in body.split(":", 1))
        if key == "wait":
            steps.append({"wait": int(value)})
        elif key in ("input", "direction"):
            steps.append({"input": value})
        elif key in ("raw", "repeat"):
            steps.append({key: json.loads(value)})
        else:
            raise ValueError(f"{path}:{number}: unknown script operation {key}")
    if not anchor or opponent is None or not in_script or not steps:
        raise ValueError(f"{path}: anchor, opponent, and non-empty script are required")
    return ScenarioScript(anchor, opponent, steps)


def token_input(token: str, facing: int) -> tuple[int, ...]:
    normalized = token.strip().upper()
    if normalized in ACTION_INPUTS:
        return ACTION_INPUTS[normalized]
    match = re.fullmatch(r"([1-9])?([ABCD])?", normalized)
    if not match or not any(match.groups()):
        raise ValueError(f"unsupported input token: {token}")
    direction = int(match.group(1) or "5")
    button = match.group(2)
    relative = {
        1: (-1, 1), 2: (0, 1), 3: (1, 1),
        4: (-1, 0), 5: (0, 0), 6: (1, 0),
        7: (-1, -1), 8: (0, -1), 9: (1, -1),
    }[direction]
    horizontal = relative[0] * (1 if facing >= 0 else -1)
    values = [horizontal, relative[1], 0, 0, 0, 0, 0, 0]
    if button:
        values[{"A": 2, "B": 3, "C": 4, "D": 5}[button]] = 1
    return tuple(values)


def compile_steps(steps: list[dict[str, Any]], facing: int) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    for step in steps:
        if "wait" in step:
            count = int(step["wait"])
            if count < 0:
                raise ValueError("wait must be non-negative")
            result.extend([NEUTRAL] * count)
        elif "input" in step:
            result.append(token_input(str(step["input"]), facing))
        elif "raw" in step:
            raw_steps = step["raw"]
            if not isinstance(raw_steps, list):
                raise ValueError("raw must be a JSON list")
            for item in raw_steps:
                if isinstance(item, str):
                    result.append(token_input(item, facing))
                elif isinstance(item, dict):
                    token = str(item["input"])
                    frames = int(item.get("frames", 1))
                    if frames <= 0:
                        raise ValueError("raw frame count must be positive")
                    result.extend([token_input(token, facing)] * frames)
                else:
                    raise ValueError("raw entries must be strings or objects")
        elif "repeat" in step:
            repeat = step["repeat"]
            if not isinstance(repeat, dict) or "times" not in repeat or "steps" not in repeat:
                raise ValueError("repeat requires JSON object with times and steps")
            times = int(repeat["times"])
            if times <= 0:
                raise ValueError("repeat times must be positive")
            nested = repeat["steps"]
            if not isinstance(nested, list):
                raise ValueError("repeat steps must be a list")
            compiled = compile_steps(nested, facing)
            result.extend(compiled * times)
        else:
            raise ValueError(f"unsupported step: {step}")
    return result


def run_script(script: ScenarioScript, pid: int) -> dict[str, object]:
    anchor = read_anchor(script.anchor)
    with BridgeClient(pid) as client:
        snapshot = client.snapshot()
        expected_hash = int(anchor["target_hash"], 16)
        if not snapshot.in_gameplay or snapshot.run_state_name != "PAUSED":
            raise RuntimeError("script target must be a paused gameplay process")
        if snapshot.game_frame != anchor["target_frame"] or snapshot.latest.stateHash != expected_hash:
            raise RuntimeError(
                f"PID {pid} is not at anchor '{script.anchor}': "
                f"frame={snapshot.game_frame} hash={snapshot.latest.stateHash:016X}"
            )
        player = snapshot.latest.p1 if script.opponent == 0 else snapshot.latest.p2
        frames = compile_steps(script.steps, player.facing)
        trace = []
        for index, controlled in enumerate(frames, 1):
            pair = InputPair(controlled, NEUTRAL) if script.opponent == 0 else InputPair(NEUTRAL, controlled)
            before = client.snapshot().game_frame
            sequence = client.step_with_inputs(pair.p1, pair.p2)
            acknowledged = client.wait_for_ack(sequence)
            if acknowledged.ack_seq != sequence:
                raise RuntimeError(f"script frame {index} was not acknowledged")
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                current = client.snapshot()
                if current.game_frame == before + 1 and current.run_state_name == "PAUSED":
                    break
                time.sleep(0.005)
            else:
                raise RuntimeError(f"script frame {index} did not advance exactly one frame")
            actor = current.latest.p1 if script.opponent == 0 else current.latest.p2
            trace.append({
                "script_frame": index,
                "simulation_frame": current.game_frame,
                "input": list(controlled),
                "effective_input": logical_to_list(actor.input),
                "action": actor.actionId,
                "sequence": actor.sequenceId,
                "subsequence": actor.subsequenceId,
                "animation_frame": actor.animationFrame,
                "hitstop": actor.hitstop,
                "objects": actor.objectCount,
                "captured_objects": current.latest.p1ObjectCount if script.opponent == 0 else current.latest.p2ObjectCount,
                "state_hash": f"{current.latest.stateHash:016X}",
            })
        final = client.snapshot()
        return {
            "anchor": script.anchor,
            "pid": pid,
            "opponent": f"p{script.opponent + 1}",
            "frames": len(frames),
            "start_frame": anchor["target_frame"],
            "final_frame": final.game_frame,
            "final_hash": f"{final.latest.stateHash:016X}",
            "max_objects": max((item["captured_objects"] for item in trace), default=0),
            "trace": trace,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SokuRL deterministic ScenarioRunner v1")
    commands = parser.add_subparsers(dest="command", required=True)
    anchor = commands.add_parser("anchor")
    anchor_commands = anchor.add_subparsers(dest="anchor_command", required=True)
    save = anchor_commands.add_parser("save")
    save.add_argument("name")
    save.add_argument("--pid", type=int, required=True)
    load = anchor_commands.add_parser("load")
    load.add_argument("name")
    load.add_argument("--pid", type=int)
    script = commands.add_parser("script")
    script_commands = script.add_subparsers(dest="script_command", required=True)
    run = script_commands.add_parser("run")
    run.add_argument("path", type=Path)
    run.add_argument("--pid", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "anchor" and args.anchor_command == "save":
            document = save_anchor(args.name, args.pid)
            print(json.dumps({
                "anchor": args.name,
                "path": str(anchor_path(args.name)),
                "target_frame": document["target_frame"],
                "target_hash": document["target_hash"],
            }, indent=2))
            return 0
        if args.command == "anchor" and args.anchor_command == "load":
            if args.pid is not None:
                sokurl.shutdown(5.0, args.pid)
            instance = load_anchor(args.name)
            pid = instance.pid
            snapshot = instance.client.snapshot()
            instance.client.close()
            print(json.dumps({
                "anchor": args.name,
                "pid": pid,
                "frame": snapshot.game_frame,
                "hash": f"{snapshot.latest.stateHash:016X}",
                "state": snapshot.run_state_name,
            }, indent=2))
            print("ANCHOR_READY")
            return 0
        script = parse_script(args.path)
        result = run_script(script, args.pid)
        print(json.dumps(result, indent=2))
        return 0
    except (BridgeUnavailable, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

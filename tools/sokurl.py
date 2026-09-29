from __future__ import annotations

import argparse
import configparser
import ctypes
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
from ctypes import wintypes

from bridge_shared import BridgeClient, BridgeUnavailable


ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = ROOT / "th123_jp"
GAME_EXE = GAME_DIR / "th123.exe"
SKIPINTRO_INI = GAME_DIR / "modules" / "SkipIntro" / "SkipIntro.ini"
EXPECTED_MD5 = "DF35D1FBC7B583317ADABE8CD9F53B2E"
MAPPING_NAME_FORMAT = r"Local\SokuRLBridge_{}"
CONTROL_MAGIC = 0x554B4F53

SCENE_ADDRESS = 0x008A0044
BATTLE_MODE_ADDRESS = 0x00898690
P1_CHARACTER_ADDRESS = 0x00899D10
P2_CHARACTER_ADDRESS = 0x00899D30
CURRENT_SCENE_ADDRESS = 0x008A000C
LEFT_SELECTION_STAGE_OFFSET = 0x22C0
RIGHT_SELECTION_STAGE_OFFSET = 0x22C1
SCENE_SELECT = 3
SCENE_BATTLE = 5
BATTLE_MODE_VSPLAYER = 3
BATTLE_MODE_PRACTICE = 8
BATTLE_SUBMODE_REPLAY = 2

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
FILE_MAP_READ = 0x0004
WM_CLOSE = 0x0010
WAIT_OBJECT_0 = 0
INFINITE = 0xFFFFFFFF

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.ReadProcessMemory.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.ReadProcessMemory.restype = wintypes.BOOL
kernel32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.OpenFileMappingW.restype = wintypes.HANDLE
kernel32.MapViewOfFile.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t,
]
kernel32.MapViewOfFile.restype = ctypes.c_void_p
kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.CreateMutexW.restype = wintypes.HANDLE
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
kernel32.ReleaseMutex.restype = wintypes.BOOL
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL


@dataclass(frozen=True)
class BridgeState:
    version: int
    connected: bool
    in_gameplay: bool
    simulation_frame: int


@dataclass(frozen=True)
class RuntimeState:
    pid: int
    bridge: BridgeState | None
    scene: int | None
    battle_mode: int | None
    p1_character: int | None
    p2_character: int | None
    p1_selection_stage: int | None
    p2_selection_stage: int | None
    health: str


def _game_processes() -> list[psutil.Process]:
    expected = str(GAME_EXE.resolve()).casefold()
    matches = []
    for process in psutil.process_iter(["pid", "exe"]):
        try:
            executable = process.info["exe"]
            if executable and str(Path(executable).resolve()).casefold() == expected:
                matches.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    return matches


def _read_bridge(pid: int) -> BridgeState | None:
    mapping = kernel32.OpenFileMappingW(FILE_MAP_READ, False, MAPPING_NAME_FORMAT.format(pid))
    if not mapping:
        return None
    view = None
    try:
        view = kernel32.MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 80)
        if not view:
            return None
        raw = ctypes.string_at(view, 80)
        magic, version, struct_size = struct.unpack_from("<III", raw)
        if magic != CONTROL_MAGIC:
            return None
        if version == 1 and struct_size == 80:
            frame = struct.unpack_from("<Q", raw, 56)[0]
            connected, in_gameplay = struct.unpack_from("<II", raw, 64)
        elif version in (2, 3) and struct_size >= 160:
            kernel32.UnmapViewOfFile(view)
            view = kernel32.MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 160)
            if not view:
                return None
            raw = ctypes.string_at(view, 160)
            frame = struct.unpack_from("<Q", raw, 112)[0]
            connected, in_gameplay = struct.unpack_from("<II", raw, 84)
        elif version in (4, 5, 6, 7) and struct_size >= 192:
            kernel32.UnmapViewOfFile(view)
            view = kernel32.MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 192)
            if not view:
                return None
            raw = ctypes.string_at(view, 192)
            frame = struct.unpack_from("<Q", raw, 144)[0]
            connected, in_gameplay = struct.unpack_from("<II", raw, 116)
        else:
            return None
        return BridgeState(version, bool(connected), bool(in_gameplay), frame)
    finally:
        if view:
            kernel32.UnmapViewOfFile(view)
        kernel32.CloseHandle(mapping)


def _read_process_values(pid: int) -> tuple[int, int, int, int, int | None, int | None]:
    handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "OpenProcess failed")
    try:
        def read_value(address: int, size: int) -> int:
            buffer = (ctypes.c_ubyte * size)()
            read = ctypes.c_size_t()
            if not kernel32.ReadProcessMemory(
                handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(read)
            ) or read.value != size:
                raise OSError(ctypes.get_last_error(), f"ReadProcessMemory failed at 0x{address:08X}")
            return int.from_bytes(bytes(buffer), "little")

        values = []
        for address, size in (
            (SCENE_ADDRESS, 4),
            (BATTLE_MODE_ADDRESS, 1),
            (P1_CHARACTER_ADDRESS, 4),
            (P2_CHARACTER_ADDRESS, 4),
        ):
            values.append(read_value(address, size))
        left_stage = right_stage = None
        if values[0] == SCENE_SELECT:
            scene_pointer = read_value(CURRENT_SCENE_ADDRESS, 4)
            if scene_pointer:
                left_stage = read_value(scene_pointer + LEFT_SELECTION_STAGE_OFFSET, 1)
                right_stage = read_value(scene_pointer + RIGHT_SELECTION_STAGE_OFFSET, 1)
        return *values, left_stage, right_stage
    finally:
        kernel32.CloseHandle(handle)


def _expected_characters() -> tuple[int, int]:
    config = configparser.ConfigParser()
    if not config.read(SKIPINTRO_INI, encoding="ascii"):
        raise RuntimeError(f"cannot read {SKIPINTRO_INI}")
    return config.getint("P1", "character"), config.getint("P2", "character")


def _runtime_state(process: psutil.Process) -> RuntimeState:
    bridge = _read_bridge(process.pid)
    try:
        scene, mode, p1, p2, left_stage, right_stage = _read_process_values(process.pid)
    except (OSError, psutil.NoSuchProcess):
        scene = mode = p1 = p2 = None
        left_stage = right_stage = None
    expected_p1, expected_p2 = _expected_characters()
    if bridge is None or not bridge.connected:
        health = "BRIDGE_DISCONNECTED"
    elif (
        mode == BATTLE_MODE_PRACTICE
        and p1 == expected_p1
        and p2 == expected_p2
        and scene == SCENE_BATTLE
        and bridge.in_gameplay
    ):
        health = "PRACTICE_READY"
    elif (
        mode == BATTLE_MODE_VSPLAYER
        and p1 == expected_p1
        and p2 == expected_p2
        and scene == SCENE_BATTLE
        and bridge.in_gameplay
    ):
        health = "VS_READY"
    elif mode == BATTLE_MODE_PRACTICE and p1 == expected_p1 and p2 == expected_p2 and scene == SCENE_SELECT:
        health = "PRACTICE_PRESET_READY"
    else:
        health = "STARTING"
    return RuntimeState(
        process.pid, bridge, scene, mode, p1, p2, left_stage, right_stage, health
    )


def _print_state(state: RuntimeState) -> None:
    payload = asdict(state)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate_game() -> None:
    digest = hashlib.md5(GAME_EXE.read_bytes()).hexdigest().upper()
    if digest != EXPECTED_MD5:
        raise RuntimeError(f"th123.exe MD5 mismatch: {digest}")


def practice(timeout: float, pid: int | None) -> int:
    _validate_game()
    if pid is None:
        process = psutil.Process(subprocess.Popen([str(GAME_EXE)], cwd=GAME_DIR).pid)
        print(f"launched {GAME_EXE} (PID {process.pid})")
    else:
        process = _select_process(pid)
        print(f"attached to {GAME_EXE} (PID {process.pid})")

    deadline = time.monotonic() + timeout
    last_state = None
    client = None
    confirmations = 0
    next_confirmation = 0.0
    while time.monotonic() < deadline:
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError(f"th123 exited before Practice became ready (PID {process.pid})")
        last_state = _runtime_state(process)
        if last_state.health == "PRACTICE_READY":
            _print_state(last_state)
            print(f"PRACTICE_READY confirmations={confirmations}")
            if client:
                client.close()
            return 0
        if last_state.health == "PRACTICE_PRESET_READY" and time.monotonic() >= next_confirmation:
            if client is None:
                client = BridgeClient(process.pid)
            if confirmations >= 12:
                raise RuntimeError("menu confirmation limit reached before battle")
            sequence = client.menu_confirm()
            acknowledged = client.wait_for_ack(sequence, timeout=2.0)
            if acknowledged.ack_seq != sequence:
                raise RuntimeError(f"menu confirmation {confirmations + 1} was not acknowledged")
            confirmations += 1
            next_confirmation = time.monotonic() + 0.6
        time.sleep(0.1)
    if client:
        client.close()
    if last_state:
        _print_state(last_state)
    raise RuntimeError(f"timed out after {timeout:.1f}s waiting for PRACTICE_READY")


def _launch_vs_from_title(
    timeout: float,
    *,
    headless: bool = False,
    unlimited: bool = False,
    seed: int | None = None,
    pause_at_start: bool = False,
) -> psutil.Process:
    if unlimited and not headless:
        raise ValueError("--unlimited requires --headless")
    config = configparser.ConfigParser()
    if not config.read(SKIPINTRO_INI, encoding="ascii"):
        raise RuntimeError(f"cannot read {SKIPINTRO_INI}")
    env = os.environ.copy()
    env.update({
        "SOKURL_VS_BOOTSTRAP": "1",
        "SOKURL_VS_P1_CHARACTER": str(config.getint("P1", "character")),
        "SOKURL_VS_P2_CHARACTER": str(config.getint("P2", "character")),
        "SOKURL_VS_P1_PALETTE": str(config.getint("P1", "palette")),
        "SOKURL_VS_P2_PALETTE": str(config.getint("P2", "palette")),
        "SOKURL_VS_P1_DECK": str(config.getint("P1", "deck")),
        "SOKURL_VS_P2_DECK": str(config.getint("P2", "deck")),
        "SOKURL_VS_STAGE": "0",
        "SOKURL_VS_MUSIC": "0",
        "SOKURL_HEADLESS_RENDER": "1" if headless else "0",
        "SOKURL_UNLIMITED_PACING": "1" if unlimited else "0",
        "SOKURL_VS_PAUSE_AT_START": "1" if pause_at_start else "0",
    })
    if seed is not None:
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("VS seed must fit in uint32")
        env["SOKURL_VS_SEED"] = str(seed)
    else:
        env.pop("SOKURL_VS_SEED", None)

    mutex = kernel32.CreateMutexW(None, False, r"Local\SokuRLVsLaunchConfig")
    if not mutex:
        raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
    process = None
    original = b""
    try:
        if kernel32.WaitForSingleObject(mutex, INFINITE) != WAIT_OBJECT_0:
            raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
        original = SKIPINTRO_INI.read_bytes()
        title_config, replacements = re.subn(
            rb"(?m)^(\s*scene_id\s*=\s*)\d+(\s*)$", rb"\g<1>2\g<2>", original, count=1
        )
        if replacements != 1:
            raise RuntimeError("SkipIntro scene_id setting was not found")
        SKIPINTRO_INI.write_bytes(title_config)
        process = psutil.Process(subprocess.Popen([str(GAME_EXE)], cwd=GAME_DIR, env=env).pid)
        deadline = time.monotonic() + min(timeout, 10.0)
        while time.monotonic() < deadline:
            if not process.is_running():
                raise RuntimeError(f"th123 exited before Title bootstrap (PID {process.pid})")
            try:
                _, mode, _, _, _, _ = _read_process_values(process.pid)
                if mode == BATTLE_MODE_VSPLAYER:
                    return process
            except OSError:
                pass
            time.sleep(0.01)
        raise RuntimeError(f"Title bootstrap timeout for PID {process.pid}")
    finally:
        if original:
            SKIPINTRO_INI.write_bytes(original)
        kernel32.ReleaseMutex(mutex)
        kernel32.CloseHandle(mutex)


def versus(timeout: float, headless: bool = False, unlimited: bool = False) -> int:
    _validate_game()
    process = _launch_vs_from_title(timeout, headless=headless, unlimited=unlimited)
    print(f"launched {GAME_EXE} (PID {process.pid})")
    client = BridgeClient(process.pid)
    deadline = time.monotonic() + timeout
    first_frame = None
    last_state = None
    while time.monotonic() < deadline:
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            client.close()
            raise RuntimeError(f"th123 exited before VS Player became ready (PID {process.pid})")
        last_state = _runtime_state(process)
        if last_state.health == "VS_READY":
            frame = last_state.bridge.simulation_frame
            if first_frame is None:
                first_frame = frame
            elif frame > first_frame:
                _print_state(last_state)
                if unlimited:
                    print("VS_UNLIMITED_READY")
                else:
                    print("VS_HEADLESS_READY" if headless else "VS_READY")
                client.close()
                return 0
        time.sleep(0.05)
    if last_state:
        _print_state(last_state)
    client.close()
    raise RuntimeError(f"timed out after {timeout:.1f}s waiting for VS_READY")


def replay(path: Path, frame: int | None, timeout: float) -> int:
    _validate_game()
    path = path.resolve()
    if not path.is_file() or path.suffix.casefold() != ".rep":
        raise RuntimeError(f"not a replay file: {path}")
    if frame is not None and frame < 0:
        raise RuntimeError("--frame must be non-negative")

    process = psutil.Process(subprocess.Popen([str(GAME_EXE), str(path)], cwd=GAME_DIR).pid)
    client = None
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not process.is_running():
                raise RuntimeError(f"th123 exited before replay readiness (PID {process.pid})")
            try:
                client = BridgeClient(process.pid)
                break
            except BridgeUnavailable:
                time.sleep(0.05)
        else:
            raise RuntimeError(f"bridge timeout for replay PID {process.pid}")

        while time.monotonic() < deadline:
            snapshot = client.snapshot()
            if (
                snapshot.in_gameplay
                and snapshot.latest.sceneId == SCENE_BATTLE
                and snapshot.latest.battleSubMode == BATTLE_SUBMODE_REPLAY
                and snapshot.game_frame == 0
                and snapshot.run_state_name == "PAUSED"
            ):
                break
            time.sleep(0.01)
        else:
            raise RuntimeError(f"replay frame-zero timeout for PID {process.pid}")

        if frame is None:
            sequence = client.run()
            acknowledged = client.wait_for_ack(sequence)
            if acknowledged.ack_seq != sequence:
                raise RuntimeError("replay run command was not acknowledged")
            print(f"REPLAY_PLAYING pid={process.pid} file={path}")
            return 0

        remaining = frame
        while remaining:
            count = min(remaining, 10_000)
            before = client.snapshot().game_frame
            sequence = client.step(count)
            acknowledged = client.wait_for_ack(sequence, timeout=max(2.0, count / 1000))
            if acknowledged.ack_seq != sequence:
                raise RuntimeError(f"replay step was not acknowledged at frame {before}")
            expected = before + count
            step_deadline = time.monotonic() + max(5.0, count / 500)
            while time.monotonic() < step_deadline:
                snapshot = client.snapshot()
                if snapshot.game_frame == expected and snapshot.run_state_name == "PAUSED":
                    break
                time.sleep(0.005)
            else:
                raise RuntimeError(
                    f"replay seek stopped at {client.snapshot().game_frame}, expected {expected}"
                )
            remaining -= count

        snapshot = client.snapshot()
        payload = {
            "pid": process.pid,
            "file": str(path),
            "frame": snapshot.game_frame,
            "state": snapshot.run_state_name,
            "hash": f"{snapshot.latest.stateHash:016X}",
            "p1_character": snapshot.latest.p1.characterId,
            "p2_character": snapshot.latest.p2.characterId,
            "p1_objects": snapshot.latest.p1ObjectCount,
            "p2_objects": snapshot.latest.p2ObjectCount,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("REPLAY_FRAME_READY")
        return 0
    except Exception:
        if process.is_running():
            _post_close(process.pid)
        raise
    finally:
        if client is not None:
            client.close()


def _select_process(pid: int | None) -> psutil.Process:
    processes = _game_processes()
    if not processes:
        raise RuntimeError("th123 is not running")
    if pid is not None:
        for process in processes:
            if process.pid == pid:
                return process
        raise RuntimeError(f"th123 PID {pid} is not running")
    if len(processes) != 1:
        raise RuntimeError(f"--pid is required; found {len(processes)} instances")
    return processes[0]


def list_instances() -> int:
    processes = _game_processes()
    states = [asdict(_runtime_state(process)) for process in processes]
    print(json.dumps(states, indent=2, sort_keys=True))
    return 0


def status(pid: int | None) -> int:
    state = _runtime_state(_select_process(pid))
    _print_state(state)
    return 0 if state.health in {"PRACTICE_READY", "VS_READY"} else 1


def _post_close(pid: int) -> int:
    posted = 0

    @WNDENUMPROC
    def callback(window: int, _: int) -> bool:
        nonlocal posted
        window_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(window_pid))
        if window_pid.value == pid:
            user32.PostMessageW(window, WM_CLOSE, 0, 0)
            posted += 1
        return True

    user32.EnumWindows(callback, 0)
    return posted


def shutdown(timeout: float, pid: int | None) -> int:
    processes = [_select_process(pid)] if pid is not None else _game_processes()
    if not processes:
        print("th123 is not running")
        return 0
    for process in processes:
        print(f"requesting graceful shutdown for PID {process.pid}")
        _post_close(process.pid)
    gone, alive = psutil.wait_procs(processes, timeout=timeout)
    for process in alive:
        print(f"graceful timeout; terminating PID {process.pid}")
        process.terminate()
    _, alive = psutil.wait_procs(alive, timeout=2.0)
    for process in alive:
        print(f"terminate timeout; killing PID {process.pid}")
        process.kill()
    psutil.wait_procs(alive, timeout=2.0)
    print(f"shutdown complete ({len(processes)} process(es))")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thin SokuRL game launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)
    practice_parser = subparsers.add_parser("practice", help="launch the SkipIntro Practice preset")
    practice_parser.add_argument("--timeout", type=float, default=30.0)
    practice_parser.add_argument("--pid", type=int)
    vs_parser = subparsers.add_parser("vs", help="launch local VS Player through Title")
    vs_parser.add_argument("--timeout", type=float, default=30.0)
    vs_parser.add_argument("--headless", action="store_true", help="skip complex battle rendering")
    vs_parser.add_argument(
        "--unlimited", action="store_true", help="remove VS battle wall-clock pacing"
    )
    replay_parser = subparsers.add_parser("replay", help="launch a replay through ReplayDnD")
    replay_parser.add_argument("path", type=Path)
    replay_parser.add_argument("--frame", type=int)
    replay_parser.add_argument("--timeout", type=float, default=35.0)
    anchor_parser = subparsers.add_parser("anchor", help="save or load a ScenarioRunner anchor")
    anchor_commands = anchor_parser.add_subparsers(dest="anchor_command", required=True)
    anchor_save = anchor_commands.add_parser("save")
    anchor_save.add_argument("name")
    anchor_save.add_argument("--pid", type=int, required=True)
    anchor_load = anchor_commands.add_parser("load")
    anchor_load.add_argument("name")
    anchor_load.add_argument("--pid", type=int)
    script_parser = subparsers.add_parser("script", help="run a ScenarioRunner opponent script")
    script_commands = script_parser.add_subparsers(dest="script_command", required=True)
    script_run = script_commands.add_parser("run")
    script_run.add_argument("path", type=Path)
    script_run.add_argument("--pid", type=int, required=True)
    subparsers.add_parser("list", help="list all th123 instances")
    status_parser = subparsers.add_parser("status", help="show process and Practice status")
    status_parser.add_argument("--pid", type=int)
    shutdown_parser = subparsers.add_parser("shutdown", help="close th123")
    shutdown_parser.add_argument("--timeout", type=float, default=5.0)
    shutdown_parser.add_argument("--pid", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "practice":
            return practice(args.timeout, args.pid)
        if args.command == "vs":
            return versus(args.timeout, args.headless, args.unlimited)
        if args.command == "replay":
            return replay(args.path, args.frame, args.timeout)
        if args.command == "anchor":
            from scenario_runner import anchor_path, load_anchor, save_anchor
            if args.anchor_command == "save":
                document = save_anchor(args.name, args.pid)
                print(json.dumps({
                    "anchor": args.name,
                    "path": str(anchor_path(args.name)),
                    "target_frame": document["target_frame"],
                    "target_hash": document["target_hash"],
                }, indent=2))
                return 0
            if args.pid is not None:
                shutdown(5.0, args.pid)
            instance = load_anchor(args.name)
            snapshot = instance.client.snapshot()
            pid = instance.pid
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
        if args.command == "script":
            from scenario_runner import parse_script, run_script
            result = run_script(parse_script(args.path), args.pid)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "list":
            return list_instances()
        if args.command == "status":
            return status(args.pid)
        return shutdown(args.timeout, args.pid)
    except (BridgeUnavailable, RuntimeError, ValueError, OSError, psutil.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Launch the original replay engine at a paused frame zero."""
import os
import subprocess
import time

import psutil
import sokurl
from bridge_shared import BridgeClient, BridgeUnavailable
from game_runtime.stepping import PausedGame
from startup_dialogs import blocking_dialogs
from game_runtime.startup import title_configuration

SCENE_BATTLE = 5
BATTLE_MODE_VSPLAYER = 3
BATTLE_SUBMODE_REPLAY = 2


def launch_replay(replay, timeout, unlimited, observation_mode):
    if timeout <= 0 or type(unlimited) is not bool:
        raise ValueError("replay launch requires a positive timeout and an explicit pacing mode")
    replay = replay.resolve()
    if not replay.is_file() or replay.suffix.casefold() != ".rep":
        raise ValueError(f"not a replay file: {replay}")
    if observation_mode not in {"image", "state", "privileged_state", "diagnostic_state"}:
        raise ValueError("unsupported replay observation mode")
    sokurl._validate_game()
    with title_configuration(sokurl.SKIPINTRO_INI, timeout):
        return _start_replay(replay, timeout, unlimited, observation_mode)


def _start_replay(replay, timeout, unlimited, observation_mode):
    process = psutil.Process(
        subprocess.Popen([str(sokurl.GAME_EXE), str(replay)], cwd=sokurl.GAME_DIR,
                         env=os.environ | {"SOKURL_UNLIMITED_PACING": str(int(unlimited)),
                             "SOKURL_VS_BOOTSTRAP": "0",
                             "SOKURL_CAPTURE_IMAGES": {"image": "1", "state": "2",
                                 "privileged_state": "0", "diagnostic_state": "0"}[observation_mode]}).pid
    )
    client: BridgeClient | None = None
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if not process.is_running():
                raise RuntimeError(f"replay process exited during startup (PID {process.pid})")
            try:
                client = BridgeClient(process.pid)
                break
            except BridgeUnavailable:
                time.sleep(0.05)
        else:
            raise RuntimeError(f"bridge timeout for replay process {process.pid}")

        while time.monotonic() < deadline:
            snapshot = client.snapshot()
            dialogs = blocking_dialogs({process.pid})
            if dialogs:
                raise RuntimeError(f"replay startup was blocked: {dialogs}")
            if snapshot.in_gameplay and snapshot.latest.sceneId == SCENE_BATTLE:
                if snapshot.latest.battleMode != BATTLE_MODE_VSPLAYER:
                    raise RuntimeError(
                        f"unexpected replay battle mode {snapshot.latest.battleMode}"
                    )
                if snapshot.latest.battleSubMode != BATTLE_SUBMODE_REPLAY:
                    raise RuntimeError(
                        f"unexpected replay submode {snapshot.latest.battleSubMode}"
                    )
                if snapshot.game_frame != 0 or snapshot.run_state_name != "PAUSED":
                    raise RuntimeError(
                        f"replay did not stop at frame zero: frame={snapshot.game_frame} "
                        f"state={snapshot.run_state_name}"
                    )
                if not snapshot.checkpoint_valid:
                    raise RuntimeError("replay frame-zero checkpoint is not valid")
                client.drain_frames()
                return PausedGame(process, client, 0)
            time.sleep(0.01)
        values = sokurl._read_process_values(process.pid)
        raise RuntimeError(f"replay battle timeout for PID {process.pid}: "
                           f"scene={values[0]}, mode={values[1]}, characters={values[2:4]}")
    except Exception:
        if client is not None:
            client.close()
        if process.is_running():
            sokurl.shutdown(3.0, process.pid)
        raise


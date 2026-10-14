"""Fingerprint runtime code, original game files, and installed modules."""
import hashlib
import json
import ctypes
import os
from pathlib import Path
import sokurl

ROOT = Path(__file__).resolve().parents[2]

def fingerprints():
    platform = Path(os.environ.get("SOKURL_PLATFORM_ROOT", ROOT.parent))
    sources = sorted((ROOT / "src/soku_rl").rglob("*.py")) + sorted((ROOT / "tools").rglob("*.py")) + sorted((platform / "tools").rglob("*.py"))
    source_hashes = {("RL/" + p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT)
                      else "Platform/" + p.relative_to(platform).as_posix()):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    artifacts = [sokurl.GAME_DIR / name for name in
                 ("th123.exe", "th123a.dat", "th123b.dat", "th123c.dat", "d3d9.dll", "SWRSToys.ini")]
    artifacts += sorted((sokurl.GAME_DIR / "modules").rglob("*.dll"))
    artifacts += sorted((sokurl.GAME_DIR / "modules").rglob("*.ini"))
    artifacts += sorted((sokurl.GAME_DIR / "profile").glob("*.pf"))
    # Launch temporarily edits SkipIntro's scene setting under this same mutex.
    # Record stable files after that launch has restored the saved configuration.
    mutex = sokurl.kernel32.CreateMutexW(None, False, r"Local\SokuRLVsLaunchConfig")
    if not mutex:
        raise OSError(ctypes.get_last_error(), "cannot lock game artifacts")
    acquired = False
    try:
        acquired = sokurl.kernel32.WaitForSingleObject(mutex, 180000) == sokurl.WAIT_OBJECT_0
        if not acquired:
            raise RuntimeError("timed out waiting to fingerprint stable game settings")
        artifact_hashes = {str(p.relative_to(sokurl.GAME_DIR)).replace("\\", "/"):
                           hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    finally:
        if acquired:
            sokurl.kernel32.ReleaseMutex(mutex)
        sokurl.kernel32.CloseHandle(mutex)
    implementation = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    game_id = hashlib.sha256(json.dumps(artifact_hashes, sort_keys=True).encode()).hexdigest()
    return {"implementation": implementation, "game_id": game_id,
            "source_hashes": source_hashes, "artifact_hashes": artifact_hashes}

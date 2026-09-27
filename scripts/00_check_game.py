"""Verify the local game executable and report a running process if present."""

from __future__ import annotations

import argparse
import hashlib
import psutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = PROJECT_ROOT / "th123_jp" / "th123.exe"
EXPECTED_MD5 = "DF35D1FBC7B583317ADABE8CD9F53B2E"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as executable:
        for chunk in iter(lambda: executable.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def find_game_process(executable: Path) -> list[tuple[int, Path]]:
    matches = []

    # print("TARGET:", repr(str(executable)))
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            proc_exe = proc.info["exe"]

            if not proc_exe:
                continue

            proc_path = Path(proc_exe).resolve()

            if proc_path == executable:
                # print(f"{str(proc_path == executable):<10} PROC:  {repr(str(proc_path))}\n")
                matches.append((proc.info["pid"], proc_path))

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return matches



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--expected-md5", default=EXPECTED_MD5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    executable = args.exe.expanduser().resolve()
    expected_md5 = args.expected_md5.upper()

    if not executable.is_file():
        print(f"[ERROR] th123.exe not found: {executable}")
        return 2

    actual_md5 = md5_file(executable)
    supported = actual_md5 == expected_md5
    print(f"[OK] th123.exe found: {executable}")
    print(f"MD5: {actual_md5}")
    print(f"Version supported: {'Yes' if supported else 'No'}")

    processes = find_game_process(executable)
    if processes:
        for pid, path in processes:
            print("[OK] Running game process found")
            print(f"PID: {pid}")
            print(f"Process executable: {path}")
    else:
        print("[INFO] Game is not running")
        
    return 0 if supported else 1


if __name__ == "__main__":
    raise SystemExit(main())

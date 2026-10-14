"""Serialize temporary title-screen configuration across owned game launches."""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from pathlib import Path
import re
import configparser

from soku_rl.env.match import MatchConfig, PlayerSetup


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.CreateMutexW.restype = wintypes.HANDLE
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
kernel32.ReleaseMutex.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL


def configure_game(directory):
    import sokurl
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("game_directory must identify a directory")
    sokurl.GAME_DIR = root
    sokurl.GAME_EXE = root / "th123.exe"
    sokurl.SKIPINTRO_INI = root / "modules/SkipIntro/SkipIntro.ini"
    sokurl._validate_game()


def read_match(path):
    config = configparser.ConfigParser()
    config.read_string(path.read_text(encoding="ascii"))
    return MatchConfig(*(PlayerSetup(*(config.getint(player, field)
        for field in ("character", "palette", "deck"))) for player in ("P1", "P2")))


@contextmanager
def title_configuration(path, timeout):
    if timeout <= 0:
        raise ValueError("title configuration requires a positive timeout")
    mutex = kernel32.CreateMutexW(None, False, r"Local\SokuRLVsLaunchConfig")
    if not mutex:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        status = kernel32.WaitForSingleObject(mutex, int(timeout * 1000))
        if status not in (0, 0x80):  # Success or an abandoned mutex now owned by us.
            raise TimeoutError(f"cannot acquire game startup configuration: {status:#x}")
        acquired = True
        original = path.read_bytes()
        title, replacements = re.subn(
            rb"(?m)^(\s*scene_id\s*=\s*)\d+(\s*)$", rb"\g<1>2\g<2>", original, count=1)
        if replacements != 1:
            raise ValueError("SkipIntro scene_id setting was not found")
        try:
            path.write_bytes(title)
            yield
        finally:
            path.write_bytes(original)
    finally:
        if acquired:
            kernel32.ReleaseMutex(mutex)
        kernel32.CloseHandle(mutex)

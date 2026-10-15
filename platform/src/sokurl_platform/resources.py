"""Locate code, builds and protected external resources without copying them."""
import os
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PLATFORM_ROOT.parent


def game_directory():
    return Path(os.environ.get('SOKURL_GAME_DIR', WORKSPACE_ROOT / 'th123_jp')).resolve()


def third_party_directory():
    return Path(os.environ.get('SOKURL_THIRD_PARTY', WORKSPACE_ROOT / 'third_party')).resolve()


def bridge_library():
    return PLATFORM_ROOT / 'build/bridge/Release/SokuRLBridge.dll'

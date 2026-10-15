"""Dependency direction, compatibility identity and native transport contracts."""
import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_platform_sources_never_import_learning():
    forbidden = {'soku_rl', 'rl', 'torch', 'gymnasium', 'stable_baselines3'}
    for path in (ROOT / 'src/sokurl_platform').glob('*.py'):
        nodes = ast.walk(ast.parse(path.read_text(encoding='utf-8-sig')))
        for node in nodes:
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.split('.')[0] in forbidden for n in names), path


def test_isolated_platform_imports_without_rl_or_numerical_stack():
    code = (
        f'import sys;sys.path.insert(0,{str(ROOT / "src")!r});'
        'from sokurl_platform.native_batch import NativeGameBatch;'
        'from sokurl_platform.transport import WorkerConnection;'
        'from sokurl_platform.images import ImageClient;'
        'from sokurl_platform.native_reader import PrivilegedReader;'
        'from sokurl_platform.replay_recording import OriginalReplayReader;'
        'from sokurl_platform.realtime import RealtimeInput;'
        'assert not any(n.split(".")[0] in {"soku_rl","torch","gymnasium","stable_baselines3"} for n in sys.modules)'
    )
    subprocess.run([sys.executable, '-I', '-B', '-c', code], check=True, timeout=30)


def test_legacy_import_is_same_implementation_not_a_copy():
    import bridge_shared
    from sokurl_platform import bridge_shared as owned
    assert bridge_shared is owned
    assert bridge_shared.RawFrameState is owned.RawFrameState


def test_directory_name_does_not_shadow_standard_library_platform():
    import platform
    assert Path(platform.__file__).parent != ROOT
    assert platform.system() == 'Windows'

"""Explicit learning launcher over the sibling, independently usable Platform."""
import os
from pathlib import Path
import runpy
import sys

RL_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = RL_ROOT.parent
PLATFORM_ROOT = Path(os.environ.get("SOKURL_PLATFORM_ROOT", WORKSPACE_ROOT / 'platform')).resolve()

def configure():
    if not (PLATFORM_ROOT / "src/sokurl_platform/__init__.py").is_file():
        raise RuntimeError(f"Platform tools unavailable: {PLATFORM_ROOT}")
    paths = [str(RL_ROOT / "src"), str(RL_ROOT / "tools"),
             str(PLATFORM_ROOT / "src"), str(PLATFORM_ROOT / "tools")]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.environ["SOKURL_PLATFORM_ROOT"] = str(PLATFORM_ROOT)
    os.environ["SOKURL_RL_ROOT"] = str(RL_ROOT)
    os.environ["SOKURL_PROJECT_ROOT"] = str(WORKSPACE_ROOT)
    os.environ["SOKURL_PYTHON"] = sys.executable
    from sokurl_platform.resources import game_directory
    os.environ['SOKURL_GAME_DIR'] = str(game_directory())

def main():
    configure()
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python RL/run.py <tool name> [arguments]")
    name = sys.argv.pop(1)
    if Path(name).name != name:
        raise SystemExit("Tool must be a name, not a path")
    target = RL_ROOT / "tools" / (name if name.endswith(".py") else name + ".py")
    if not target.is_file():
        raise SystemExit(f"Unknown RL tool: {name}")
    os.chdir(WORKSPACE_ROOT)
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")

if __name__ == "__main__":
    main()

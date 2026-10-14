"""Explicit RL launcher; Platform implementations remain in the parent tree."""
import os
from pathlib import Path
import runpy
import sys

RL_ROOT = Path(__file__).resolve().parent
PLATFORM_ROOT = Path(os.environ.get("SOKURL_PLATFORM_ROOT", RL_ROOT.parent)).resolve()

def configure():
    if not (PLATFORM_ROOT / "tools/bridge_shared.py").is_file():
        raise RuntimeError(f"Platform tools unavailable: {PLATFORM_ROOT}")
    paths = [str(RL_ROOT / "src"), str(RL_ROOT / "tools"), str(PLATFORM_ROOT / "tools")]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.environ["SOKURL_PLATFORM_ROOT"] = str(PLATFORM_ROOT)
    os.environ["SOKURL_RL_ROOT"] = str(RL_ROOT)
    os.environ["SOKURL_PROJECT_ROOT"] = str(PLATFORM_ROOT)
    os.environ["SOKURL_PYTHON"] = sys.executable

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
    os.chdir(PLATFORM_ROOT)
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")

if __name__ == "__main__":
    main()

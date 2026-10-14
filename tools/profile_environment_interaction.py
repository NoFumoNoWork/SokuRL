"""Compatibility entry; the unique implementation lives under RL/tools."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    sys.argv.insert(1, "profile_environment_interaction")
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "RL/run.py"), run_name="__main__")

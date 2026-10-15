"""Use the same explicit roots as real learning entry points during collection."""
from pathlib import Path
import runpy


def pytest_configure(config):
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'run.py'))['configure']()

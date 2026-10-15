"""Ownership and dependency guards for the physically isolated RL tree."""
import ast
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]

def test_single_rl_package_and_platform_dependencies():
    assert not (ROOT / "src/soku_rl").exists()
    assert (ROOT / "rl/src/soku_rl/rl").is_dir()
    platform = tomllib.loads((ROOT / "platform/pyproject.toml").read_text())
    assert platform["project"]["dependencies"] == ["psutil==7.2.2"]
    for path in (ROOT / "platform/src/sokurl_platform").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any(name.split(".")[0] in {"soku_rl", "torch", "stable_baselines3"} for name in imports), path

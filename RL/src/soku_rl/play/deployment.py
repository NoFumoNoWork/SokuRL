"""Build content-addressed ONNX artifacts in a separate process before play."""
import hashlib
import json
import errno
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf


def prepare_play_candidate(candidate, settings):
    spec = candidate["policy"]
    if spec["kind"] in {"rule", "onnx", "onnx_recurrent", "onnx_dqn"}:
        return candidate
    steps = settings["verification_steps"]
    if type(steps) is not int or steps < 512:
        raise ValueError("play deployment must verify at least 512 decisions")
    spec = dict(spec)
    for key in ("path", "training_config", "model_config"):
        if key in spec:
            spec[key] = str(Path(spec[key]).resolve(strict=True))
    root = Path(__file__).resolve().parents[3]
    source_files = [root / "tools/export_policy.py", *sorted((root / "src/soku_rl/policy").glob("*.py")),
        *sorted((root / "src/soku_rl/rl").glob("*.py")), *sorted((root / "src/soku_rl/env").rglob("*.py"))]
    identity = {"schema": 1, "policy": spec, "verification_steps": steps,
        "inputs": {key: hashlib.sha256(Path(spec[key]).read_bytes()).hexdigest()
            for key in ("path", "training_config", "model_config") if key in spec},
        "export_sources": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(settings["cache"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix="build-", dir=cache))
        config_dir = staging / "config"
        config_dir.mkdir()
        # Same-parent publication also works under the Linux Landlock ABI 1
        # restriction against cross-directory renames.
        output = staging.with_name(staging.name + "-artifact")
        config = {"candidate": candidate | {"policy": spec}, "verification_steps": steps,
            "seed": 1732, "output": str(output),
            "hydra": {"run": {"dir": str(staging / "hydra")}, "output_subdir": None, "job": {"chdir": False}}}
        OmegaConf.save(OmegaConf.create(config), config_dir / "export.yaml")
        print("正在导出并校验 ONNX；此步骤在游戏启动前完成。", flush=True)
        with (staging / "export.log").open("w") as log:
            completed = subprocess.run([sys.executable, "-B", str(root / "tools/export_policy.py"),
                "--config-path", str(config_dir), "--config-name", "export"], cwd=root,
                stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"ONNX preparation failed; see {staging / 'export.log'}")
        # Do not publish an artifact built from a concurrently changing checkpoint.
        for field, digest in identity["inputs"].items():
            if hashlib.sha256(Path(spec[field]).read_bytes()).hexdigest() != digest:
                raise RuntimeError("deployment source changed during export")
        (output / "source.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
        shutil.copyfile(staging / "export.log", output / "export.log")
        try:
            output.rename(destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            # Another preparer completed the exact same immutable key first.
            shutil.rmtree(output)
        shutil.rmtree(staging)
    if json.loads((destination / "source.json").read_text()) != identity:
        raise ValueError("ONNX cache identity differs")
    return candidate | {"policy": {"kind": "onnx", "path": str(destination / "policy.json")}}

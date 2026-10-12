"""Check a staged Platform-only publication and bind its source/evidence hashes."""
import hashlib
import json
import subprocess
from pathlib import Path


root = Path.cwd()
paths = subprocess.check_output(
    ["git", "diff", "--cached", "--name-only"], text=True).splitlines()
paths = [name for name in paths if name !=
         "docs/reports/platform_main/20261013/publication_manifest.json"]
if not paths:
    raise SystemExit("BLOCKED: no staged publication")
allowed_roots = {"native", "tools", "tests", "docs"}
allowed_files = {"README.md", "AGENTS.md", ".gitignore"}
for name in paths:
    path = Path(name)
    if path.parts[0] not in allowed_roots and name not in allowed_files:
        raise SystemExit(f"FAIL: non-platform publication path {name}")
    if any(part in {"ABORTED", "SokuSim", "checkpoints", "models"} for part in path.parts):
        raise SystemExit(f"FAIL: excluded asset {name}")
    if path.suffix.lower() in {".dll", ".exe", ".dat", ".bin", ".raw", ".pt", ".pth"}:
        raise SystemExit(f"FAIL: binary or training artifact {name}")
    if path.name in {"train.py", "rollout_worker.py", "game_batch.py", "worker_pipe.py"}:
        raise SystemExit(f"FAIL: RL worker/trainer file {name}")
records = []
for name in paths:
    data = (root / name).read_bytes()
    records.append({"path": name, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()})
report = {"status": "PASS", "base": subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True).strip(), "files": records,
    "scope": "Platform only; no src/config/training/GPU/binary changes"}
output = root / "docs/reports/platform_main/20261013/publication_manifest.json"
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"PASS: {len(records)} staged Platform paths; manifest {output}")

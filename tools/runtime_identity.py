"""Fingerprint runtime code, original game files, and installed modules."""
import hashlib
import json
from pathlib import Path
import sokurl

ROOT = Path(__file__).resolve().parents[1]

def fingerprints():
    sources = sorted((ROOT / "src/soku_rl").rglob("*.py")) + [
        ROOT / "tools" / name for name in (
            "game_batch.py", "evaluate.py", "bridge_shared.py", "sokurl.py", "runtime_identity.py", "rollout_worker.py",
            "image_shared.py", "headless_validation.py", "interaction_benchmark.py", "unlimited_benchmark.py")]
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    artifacts = [sokurl.GAME_DIR / name for name in
                 ("th123.exe", "th123a.dat", "th123b.dat", "th123c.dat", "d3d9.dll", "SWRSToys.ini")]
    artifacts += sorted((sokurl.GAME_DIR / "modules").rglob("*.dll"))
    artifacts += sorted((sokurl.GAME_DIR / "modules").rglob("*.ini"))
    artifacts += sorted((sokurl.GAME_DIR / "profile").glob("*.pf"))
    artifact_hashes = {str(p.relative_to(sokurl.GAME_DIR)).replace("\\", "/"):
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    implementation = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    game_id = hashlib.sha256(json.dumps(artifact_hashes, sort_keys=True).encode()).hexdigest()
    return {"implementation": implementation, "game_id": game_id,
            "source_hashes": source_hashes, "artifact_hashes": artifact_hashes}

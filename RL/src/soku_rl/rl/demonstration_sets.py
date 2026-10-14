"""Aggregate compatible, disjoint expert-labelled datasets without moving held-out games."""
import hashlib
from pathlib import Path

from soku_rl.rl.behavior_cloning import load_demonstrations
from soku_rl.rl.demonstration_supervision import supervision_mask


def load_demonstration_sets(sources, interface, value_coef):
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError("at least one demonstration dataset is required")
    paths = [Path(source).resolve(strict=True) for source in sources]
    if len(set(paths)) != len(paths):
        raise ValueError("demonstration datasets must be distinct")
    samples = {"train": [], "validation": []}
    identities, worlds = [], set()
    for path in paths:
        rows, manifest, contract, digest = load_demonstrations(path, interface)
        control = "teacher" if manifest["schema"] == 1 else manifest["control"]
        if control in {"learner", "teacher_takeover", "teacher_noise"} and value_coef != 0:
            raise ValueError("teacher labels with a different behavior policy require value_coef=0; returns belong to the behavior policy")
        current = {row["world_seed"] for row in manifest["episodes"]}
        if current & worlds:
            raise ValueError("demonstration datasets overlap world seeds or held-out episodes")
        worlds.update(current)
        signature = {"teacher_fingerprint": manifest["teacher_fingerprint"],
            "teacher": contract["teacher"],
            "matchups": contract["algorithm"]["matchups"],
            "opponents": contract["algorithm"]["opponents"],
            "reserved": {split: contract["excluded"][split] for split in ("validation", "test")}}
        if not identities:
            reference, training = signature, contract
        elif signature != reference:
            raise ValueError("aggregated demonstrations must share teacher, matchups, opponents and evaluation exclusions")
        for split in samples:
            samples[split].extend(rows[split])
        identities.append({"dataset": str(path), "manifest_sha256": digest,
            "config_sha256": hashlib.sha256((path / "config.yaml").read_bytes()).hexdigest(),
            "teacher_fingerprint": manifest["teacher_fingerprint"], "control": control,
            "behavior_fingerprint": manifest["teacher_fingerprint"] if manifest["schema"] == 1
                else manifest["behavior_fingerprint"],
            "environment_steps": manifest["successful_env_steps"],
            "frames": {split: len(rows[split]) for split in samples},
            "worlds": {split: [row["world_seed"] for row in manifest["episodes"] if row["split"] == split]
                for split in samples}})
        if manifest["schema"] == 3:
            identities[-1]["supervised_frames"] = {
                split: int(supervision_mask(rows[split]).sum()) for split in samples}
    return samples, training, identities

"""One CPU-only entry for exported actors and episode-stable populations."""
import hashlib
import json
from pathlib import Path

from soku_rl.policy.contract import read_training_contract
from soku_rl.policy.population import MixturePolicy, UniformPolicy


def load_portable(name, manifest_path, interface):
    path = Path(manifest_path).resolve(strict=True)
    data = path.read_bytes()
    manifest = json.loads(data)
    if manifest["format"] in {"sokurl-recurrent-onnx-v1", "sokurl-actor-onnx-v1"}:
        from soku_rl.policy.onnx import OnnxPolicy
        return OnnxPolicy(name, path, interface)
    if manifest["format"] == "sokurl-dqn-onnx-v1":
        from soku_rl.policy.onnx_dqn import OnnxDQNPolicy
        return OnnxDQNPolicy(name, path, interface)
    if manifest["format"] != "sokurl-mixture-onnx-v1":
        raise ValueError("unsupported portable policy format")
    training = path.parent / manifest["training_config"]
    read_training_contract(training, interface)
    if hashlib.sha256(training.read_bytes()).hexdigest() != manifest["training_sha256"]:
        raise ValueError("portable mixture training checksum differs")
    members = []
    for entry in manifest["members"]:
        if entry["kind"] == "uniform":
            if entry["num_actions"] != interface.action_space.n:
                raise ValueError("portable uniform action space differs")
            members.append(UniformPolicy(entry["name"], entry["num_actions"]))
        elif entry["kind"] == "onnx":
            child = path.parent / entry["path"]
            if hashlib.sha256(child.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("portable member checksum differs")
            members.append(load_portable(entry["name"], child, interface))
        else:
            raise ValueError("portable mixtures require ONNX or uniform members")
    return MixturePolicy(name, members, manifest["probabilities"], hashlib.sha256(data).hexdigest())

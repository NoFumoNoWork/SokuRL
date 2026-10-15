"""Deploy greedy Q values on CPU without importing the training framework."""
import hashlib
import json
from pathlib import Path

import numpy as np

from soku_rl.policy.base import RLPolicy
from soku_rl.policy.contract import read_training_contract


class OnnxDQNPolicy(RLPolicy):
    def __init__(self, name, manifest_path, interface):
        import onnxruntime as ort

        path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["format"] != "sokurl-dqn-onnx-v1":
            raise ValueError("unsupported DQN CPU policy format")
        read_training_contract(path.parent / manifest["training_config"], interface)
        if "training_sha256" in manifest and hashlib.sha256(
                (path.parent / manifest["training_config"]).read_bytes()).hexdigest() != manifest["training_sha256"]:
            raise ValueError("CPU training contract checksum differs")
        model = path.parent / manifest["model"]
        self.fingerprint = hashlib.sha256(model.read_bytes()).hexdigest()
        if self.fingerprint != manifest["model_sha256"]:
            raise ValueError("CPU model checksum differs from its manifest")
        self.shape = tuple(manifest["observation_shape"])
        self.num_actions = manifest["num_actions"]
        if self.shape != interface.observation_space.shape or self.num_actions != interface.action_space.n:
            raise ValueError("CPU policy observation or action space differs")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.session.disable_fallback()
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "observation" or inputs[0].shape != [1, *self.shape]
                or inputs[0].type != "tensor(float)" or len(outputs) != 1
                or outputs[0].name != "q_values" or outputs[0].shape != [1, self.num_actions]
                or outputs[0].type != "tensor(float)"):
            raise ValueError("CPU DQN input/output schema differs")
        self.name, self.manifest = name, manifest

    def spawn(self, seed):
        if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF:
            raise ValueError("CPU policy requires a supported uint32 seed")
        return OnnxDQNEpisode(self)


class OnnxDQNEpisode:
    def __init__(self, policy):
        self.policy = policy

    def act(self, observation):
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != self.policy.shape or not np.isfinite(values).all():
            raise ValueError("observation does not match the CPU policy")
        q_values, = self.policy.session.run(None, {"observation": values[None]})
        if q_values.shape != (1, self.policy.num_actions) or not np.isfinite(q_values).all():
            raise RuntimeError("CPU DQN produced invalid Q values")
        return int(q_values[0].argmax())

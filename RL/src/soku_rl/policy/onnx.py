"""Run a verified recurrent actor on one CPU thread without importing PyTorch."""
import hashlib
import json
from pathlib import Path

import numpy as np

from soku_rl.policy.contract import read_training_contract
from soku_rl.policy.base import RLPolicy


class OnnxPolicy(RLPolicy):
    def __init__(self, name, manifest_path, interface):
        import onnxruntime as ort

        path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["format"] not in {"sokurl-recurrent-onnx-v1", "sokurl-actor-onnx-v1"}:
            raise ValueError("unsupported CPU policy format")
        read_training_contract(path.parent / manifest["training_config"], interface)
        if "training_sha256" in manifest and hashlib.sha256(
                (path.parent / manifest["training_config"]).read_bytes()).hexdigest() != manifest["training_sha256"]:
            raise ValueError("CPU training contract checksum differs")
        model = path.parent / manifest["model"]
        self.fingerprint = hashlib.sha256(model.read_bytes()).hexdigest()
        if self.fingerprint != manifest["model_sha256"]:
            raise ValueError("CPU model checksum differs from its manifest")
        self.shape = tuple(manifest["observation_shape"])
        self.state_shape = tuple(manifest["state_shape"])
        self.num_actions = manifest["num_actions"]
        if self.shape != interface.observation_space.shape or self.num_actions != interface.action_space.n:
            raise ValueError("CPU policy observation or action space differs")
        recurrent = manifest["format"] == "sokurl-recurrent-onnx-v1"
        if recurrent and (len(self.state_shape) != 3 or self.state_shape[1] != 1
                or any(type(size) is not int or size < 1 for size in self.state_shape)):
            raise ValueError("CPU policy requires explicit single-player recurrent states")
        if not recurrent and self.state_shape:
            raise ValueError("feedforward CPU policy cannot have recurrent states")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.session.disable_fallback()
        expected = {"observation": [1, *self.shape]}
        outputs = {"probabilities": [1, self.num_actions]}
        if recurrent:
            expected.update(hidden=list(self.state_shape), cell=list(self.state_shape))
            outputs.update(hidden_out=list(self.state_shape), cell_out=list(self.state_shape))
        actual = {field.name: field.shape for field in self.session.get_inputs()}
        if actual != expected or any(field.type != "tensor(float)" for field in self.session.get_inputs()):
            raise ValueError("CPU model input schema differs")
        if ({field.name: field.shape for field in self.session.get_outputs()} != outputs
                or [field.name for field in self.session.get_outputs()] != list(outputs)
                or any(field.type != "tensor(float)" for field in self.session.get_outputs())):
            raise ValueError("CPU model output schema differs")
        self.name, self.manifest = name, manifest

    def spawn(self, seed):
        if type(seed) is not int or not 0 <= seed < 0xFFFFFFFF:
            raise ValueError("CPU policy requires a supported uint32 seed")
        return OnnxEpisode(self, seed)


class OnnxEpisode:
    def __init__(self, policy, seed):
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.hidden = np.zeros(policy.state_shape, np.float32)
        self.cell = np.zeros(policy.state_shape, np.float32)

    def act(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != self.policy.shape or not np.isfinite(observation).all():
            raise ValueError("observation does not match the CPU policy")
        inputs = {"observation": observation[None]}
        if self.policy.state_shape:
            inputs.update(hidden=self.hidden, cell=self.cell)
        outputs = self.policy.session.run(None, inputs)
        probabilities = outputs[0]
        hidden, cell = outputs[1:] if self.policy.state_shape else (self.hidden, self.cell)
        if (probabilities.shape != (1, self.policy.num_actions)
                or hidden.shape != self.policy.state_shape or cell.shape != self.policy.state_shape
                or not all(np.isfinite(value).all() for value in (probabilities, hidden, cell))):
            raise RuntimeError("CPU policy produced invalid outputs")
        values = probabilities[0].astype(np.float64)
        if (values < 0).any() or not np.isclose(values.sum(), 1., atol=1e-5):
            raise RuntimeError("CPU policy produced invalid probabilities")
        self.hidden, self.cell = hidden, cell
        return int(self.rng.choice(self.policy.num_actions, p=values/values.sum()))

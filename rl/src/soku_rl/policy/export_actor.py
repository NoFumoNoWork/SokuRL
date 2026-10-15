"""Export shared learned policies once, before entering the real-time play loop."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from soku_rl.env import EpisodeConfig
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.policy.loader import load_policy
from soku_rl.policy.population import MixturePolicy, PPOPolicy, UniformPolicy
from soku_rl.policy.verification_inputs import verification_observation
from soku_rl.rl.facing_policy import FacingRecurrentActorCriticPolicy, left_facing
from soku_rl.rl.recurrent_persistent_policy import PersistentRecurrentActorCriticPolicy, previous_commands

EXPORT_VERSION = 1


def encode_export_objects(features, objects, present):
    # A discarded sentinel avoids ONNX's ambiguous zero-row reshape/Gemm case.
    selected = torch.cat((objects[present], objects.new_zeros((1, objects.shape[-1]))), dim=0)
    values = features.object_encoder(features.numeric_features(selected))[:-1]
    encoded = values.new_zeros((*objects.shape[:2], values.shape[-1]))
    encoded[present] = values
    return encoded


class CategoricalActor(nn.Module):
    def __init__(self, policy, recurrent):
        super().__init__()
        self.features = copy.deepcopy(policy.pi_features_extractor)
        from soku_rl.rl.features import PrivilegedFeatures
        if isinstance(self.features, PrivilegedFeatures):
            # Evaluate present objects only; scatter back to all original slots.
            # ONNX NonZero/Gather/Scatter retain a dynamic count, including zero.
            self.features.encode_objects = MethodType(encode_export_objects, self.features)
        self.policy_net = policy.mlp_extractor
        self.action_net = policy.action_net
        self.recurrent = recurrent
        self.facing_actions = isinstance(policy, FacingRecurrentActorCriticPolicy)
        self.persistent_actions = isinstance(policy, PersistentRecurrentActorCriticPolicy)
        if self.persistent_actions:
            self.repeat_gate = policy.repeat_gate
            self.register_buffer("button_weights", policy.button_weights.clone())
        if self.facing_actions:
            self.facing_index = policy.facing_index
            self.register_buffer("action_indices", policy.facing_commands.clone())
            self.register_buffer("mirrored_indices", policy.mirrored_commands.clone())
        if recurrent:
            self.memory = policy.lstm_actor

    def forward(self, observation, *states):
        features = self.features(observation)
        if self.recurrent:
            features, states = self.memory(features.unsqueeze(0), states)
            features = features.squeeze(0)
        latent = self.policy_net.forward_actor(features)
        probabilities = self.action_net(latent).softmax(-1)
        if self.persistent_actions:
            gate = self.repeat_gate(latent).sigmoid()
            previous = previous_commands(observation, self.button_weights)
            probabilities = ((1 - gate) * probabilities).scatter_add(1, previous[:, None], gate)
        if self.facing_actions:
            permutation = torch.where(left_facing(observation, self.facing_index)[:, None],
                                      self.mirrored_indices, self.action_indices)
            probabilities = probabilities.gather(1, permutation)
        return (probabilities, *states) if self.recurrent else probabilities


class NetworkActor(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, observation):
        return self.network(observation).softmax(-1)


def export_leaf(loaded, interface, training, directory, steps, seed):
    import onnx
    import onnxruntime as ort
    from sb3_contrib import RecurrentPPO
    from soku_rl.policy.checkpoint import NetworkPolicy
    from soku_rl.policy.dqn import DQNPolicy
    directory.mkdir(parents=True, exist_ok=False)
    recurrent, q_values = False, isinstance(loaded, DQNPolicy)
    if q_values:
        actor = copy.deepcopy(loaded.model.q_net).eval()
        from soku_rl.rl.features import PrivilegedFeatures
        features = actor.features_extractor
        if isinstance(features, PrivilegedFeatures):
            features.encode_objects = MethodType(encode_export_objects, features)
    elif isinstance(loaded, PPOPolicy):
        recurrent = isinstance(loaded.model, RecurrentPPO)
        actor = CategoricalActor(loaded.model.policy, recurrent).eval()
    elif isinstance(loaded, NetworkPolicy):
        actor = NetworkActor(loaded.network).eval()
    else:
        raise TypeError(f"unsupported export policy: {type(loaded).__name__}")
    state_shape = tuple(loaded.model.policy.lstm_hidden_state_shape) if recurrent else ()
    shape = interface.observation_space.shape
    inputs = [torch.zeros((1, *shape))]
    input_names, output_names = ["observation"], ["q_values" if q_values else "probabilities"]
    if recurrent:
        inputs.extend([torch.zeros(state_shape), torch.zeros(state_shape)])
        input_names.extend(["hidden", "cell"])
        output_names.extend(["hidden_out", "cell_out"])
    path = directory / "actor.onnx"
    torch.onnx.export(actor, tuple(inputs), path, dynamo=False, opset_version=17,
        input_names=input_names, output_names=output_names)
    onnx.checker.check_model(str(path), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    errors = np.zeros(len(output_names))
    # Recurrent cell values accumulate float32 arithmetic differences near zero;
    # action probabilities retain the stricter deployment tolerance.
    absolute_tolerances = [2e-6, 1e-5, 5e-5] if recurrent else [2e-6]
    rng = np.random.default_rng(seed)
    with torch.inference_mode():
        for step in range(steps):
            if step % 128 == 0:
                states = (torch.zeros(state_shape), torch.zeros(state_shape)) if recurrent else ()
                portable = tuple(value.numpy().copy() for value in states)
            observation = verification_observation(interface, rng, step)[None]
            tensor = torch.from_numpy(observation)
            if q_values:
                expected = (loaded.model.q_net(tensor).numpy(),)
            elif isinstance(loaded, NetworkPolicy):
                expected = (loaded.network(tensor).softmax(-1).numpy(),)
            else:
                if recurrent:
                    distribution, states = loaded.model.policy.get_distribution(tensor, states, torch.zeros(1))
                else:
                    distribution = loaded.model.policy.get_distribution(tensor)
                expected = (distribution.distribution.probs.numpy(), *(v.numpy() for v in states))
            actual = session.run(None, dict(zip(input_names, (observation, *portable), strict=True)))
            for index, (reference, value) in enumerate(zip(expected, actual, strict=True)):
                np.testing.assert_allclose(value, reference, rtol=1e-4, atol=absolute_tolerances[index])
                errors[index] = max(errors[index], float(np.max(np.abs(value - reference))))
            if q_values and not np.array_equal(actual[0].argmax(-1), expected[0].argmax(-1)):
                raise RuntimeError("portable DQN changed a greedy action")
            portable = tuple(actual[1:])
    OmegaConf.save(OmegaConf.create(training), directory / "training.yaml")
    manifest = {"format": "sokurl-dqn-onnx-v1" if q_values else
        "sokurl-recurrent-onnx-v1" if recurrent else "sokurl-actor-onnx-v1",
        "export_version": EXPORT_VERSION, "model": path.name,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": loaded.fingerprint, "training_config": "training.yaml",
        "training_sha256": hashlib.sha256((directory / "training.yaml").read_bytes()).hexdigest(),
        "observation_shape": list(shape), "state_shape": list(state_shape),
        "num_actions": int(interface.action_space.n),
        "inference": "greedy_online_q" if q_values else "categorical_sampling",
        "verification": {"steps": steps, "round_resets": (steps + 127) // 128,
            "relative_tolerance": 1e-4, "absolute_tolerances": absolute_tolerances,
            "maximum_absolute_errors": errors.tolist(), "maximum_absolute_error": float(errors[0]),
            **({"greedy_action_mismatches": 0} if q_values else {})},
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}}
    (directory / "policy.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_tree(loaded, interface, training, directory, steps, seed):
    if not isinstance(loaded, MixturePolicy):
        return export_leaf(loaded, interface, training, directory, steps, seed)
    directory.mkdir(parents=True, exist_ok=False)
    members = []
    for index, member in enumerate(loaded.members):
        if isinstance(member, UniformPolicy):
            members.append({"kind": "uniform", "name": member.name, "num_actions": int(member.num_actions)})
        else:
            child = directory / f"member-{index}"
            export_tree(member, interface, training, child, steps, seed + index)
            path = child / "policy.json"
            members.append({"kind": "onnx", "name": member.name, "path": str(path.relative_to(directory)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    OmegaConf.save(OmegaConf.create(training), directory / "training.yaml")
    manifest = {"format": "sokurl-mixture-onnx-v1", "export_version": EXPORT_VERSION,
        "source_sha256": loaded.fingerprint, "training_config": "training.yaml",
        "training_sha256": hashlib.sha256((directory / "training.yaml").read_bytes()).hexdigest(),
        "members": members, "probabilities": loaded.probabilities.tolist()}
    (directory / "policy.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_actor(config):
    from soku_rl.policy.sb3_artifact import load_sb3_artifact
    from soku_rl.policy.dqn import DQNPolicy
    from soku_rl.policy.recurrent import RecurrentPPOPolicy
    spec = config["candidate"]["policy"]
    training = OmegaConf.to_container(OmegaConf.load(spec["training_config"]), resolve=True)
    interface = LearningInterface(EpisodeConfig.from_dict(training["episode"]),
        LearningConfig(**training["wrappers"]))
    if interface.episode.observation_mode not in {"state", "diagnostic_state", "privileged_state"}:
        raise ValueError("ONNX play export requires numeric state observations")
    steps = config["verification_steps"]
    if type(steps) is not int or steps < 512:
        raise ValueError("verify at least 512 sequential decisions")
    torch.set_num_threads(1)
    if spec["kind"] in {"checkpoint", "sb3", "sb3_dqn", "sb3_recurrent"}:
        path = Path(spec["path"]).resolve(strict=True)
        data = path.read_bytes()
        model, kind = load_sb3_artifact(data, "cpu")
        if model.observation_space != interface.observation_space or model.action_space != interface.action_space:
            raise ValueError("checkpoint spaces disagree with the training contract")
        cls = {"sb3": PPOPolicy, "sb3_dqn": DQNPolicy, "sb3_recurrent": RecurrentPPOPolicy}[kind]
        loaded = cls(config["candidate"]["name"], model, path)
        if loaded.fingerprint != hashlib.sha256(data).hexdigest():
            raise RuntimeError("checkpoint changed during export")
    else:
        loaded = load_policy(config["candidate"]["name"], spec, interface, "cpu")
    directory = Path(config["output"]).resolve()
    manifest = export_tree(loaded, interface, training, directory, steps, config["seed"])
    OmegaConf.save(OmegaConf.create(config | {"episode": asdict(interface.episode),
        "wrappers": asdict(interface.config)}), directory / "export.yaml")
    return manifest

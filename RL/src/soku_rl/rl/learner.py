"""Algorithm-independent creation and checkpoint lifecycle for MARL schedulers."""
import hashlib
import json
import math
from pathlib import Path

from gymnasium import spaces
from hydra.utils import get_class
from torch import nn

from soku_rl.policy.contract import read_training_contract
from soku_rl.rl import validate_payoff
from soku_rl.rl.ppo import create_ppo, parameter_hash, snapshot as ppo_snapshot


def learner_kind(config):
    # Missing means historical PPO configuration, not a device/algorithm fallback.
    kind = config.get("learner", "ppo")
    if kind not in {"ppo", "dqn"}:
        raise ValueError("rl.learner must be ppo or dqn")
    if kind == "dqn" and any(option in config for option in ("rehearsal", "online_anchor", "online_teacher")):
        raise ValueError("rehearsal, online_anchor and online_teacher require the PPO learner")
    if kind == "dqn" and "freeze_actor_representation" in config.get("ppo", {}):
        raise ValueError("freeze_actor_representation requires the PPO learner")
    return kind


def artifact_kind(config):
    if learner_kind(config) == "dqn":
        return "sb3_dqn"
    return {"mlp": "sb3", "lstm": "sb3_recurrent"}[config["policy_type"]]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_dqn(config):
    positive = ("buffer_size", "batch_size", "n_steps", "train_freq", "target_update_interval", "exploration_decay_steps")
    if any(type(config[key]) is not int or config[key] < 1 for key in positive):
        raise ValueError("DQN replay, batch, step and exploration budgets must be positive integers")
    if type(config["learning_starts"]) is not int or config["learning_starts"] < 0:
        raise ValueError("DQN learning_starts must be nonnegative")
    if type(config["gradient_steps"]) is not int or config["gradient_steps"] not in {-1} and config["gradient_steps"] < 1:
        raise ValueError("DQN gradient_steps must be positive or -1")
    for key in ("learning_rate", "max_grad_norm", "tau"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"DQN {key} must be finite and positive")
    for key in ("gamma", "exploration_initial_eps", "exploration_final_eps"):
        if not math.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f"DQN {key} must be in [0, 1]")
    if config["tau"] > 1 or config["exploration_final_eps"] > config["exploration_initial_eps"]:
        raise ValueError("invalid DQN target averaging or exploration schedule")


def save_checkpoint(model, path):
    from soku_rl.rl.dqn import DoubleDQN
    model.save(path)
    if isinstance(model, DoubleDQN):
        replay = Path(path).with_suffix(".replay.pkl")
        model.save_replay_buffer(replay)
        manifest = {"model_sha256": file_hash(path), "replay_sha256": file_hash(replay),
                    "steps": model.num_timesteps, "updates": model._n_updates}
        Path(path).with_suffix(".replay.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def snapshot(name, model, path):
    from soku_rl.rl.dqn import DoubleDQN
    from soku_rl.policy.dqn import DQNPolicy
    if isinstance(model, DoubleDQN):
        model.save(path)
        return DQNPolicy(name, model, Path(path))
    return ppo_snapshot(name, model, path)


def create_learner(env, interface, config, source, device, seed):
    if learner_kind(config) == "ppo":
        return create_ppo(env, interface, config, source, device, seed)
    from soku_rl.rl.dqn import DoubleDQN
    from soku_rl.rl.replay import PackedNStepReplayBuffer
    validate_payoff(interface, config)
    if config["policy_type"] != "mlp":
        raise ValueError("DQN requires a feedforward policy; frame history remains available")
    validate_dqn(config["dqn"])
    parameters = dict(config["dqn"])
    exploration_steps = parameters.pop("exploration_decay_steps")
    architecture = dict(parameters["policy_kwargs"])
    for key in ("features_extractor_class", "activation_fn"):
        if key in architecture and isinstance(architecture[key], str):
            architecture[key] = get_class(architecture[key])
    architecture.setdefault("activation_fn", nn.Tanh)
    parameters["policy_kwargs"] = architecture
    policy = ("MultiInputPolicy" if isinstance(env.observation_space, spaces.Dict) else
              "CnnPolicy" if len(env.observation_space.shape) == 3 else "MlpPolicy")
    parameters["replay_buffer_class"] = PackedNStepReplayBuffer
    parameters["replay_buffer_kwargs"] = {"n_steps": parameters["n_steps"],
        "gamma": parameters["gamma"], "handle_timeout_termination": False}
    if source == {"kind": "fresh"}:
        model = DoubleDQN(policy, env, seed=seed, device=device, **parameters)
        model.exploration_decay_steps = exploration_steps
        return model, source
    if set(source) != {"kind", "path", "training_config"} or source["kind"] not in {"checkpoint", "weights"}:
        raise ValueError("initial policy must be fresh, a training checkpoint, or policy weights")
    training = read_training_contract(source["training_config"], interface)
    previous = training.get("rl", training["algorithm"])
    if learner_kind(previous) != "dqn" or any(previous[k] != config[k] for k in ("policy_type", "timeout_payoff")):
        raise ValueError("DQN initialization requires the same learner and payoff")
    if source["kind"] == "checkpoint" and previous["dqn"] != config["dqn"]:
        raise ValueError("continued DQN must retain algorithm and optimizer configuration")
    if previous["dqn"]["policy_kwargs"] != config["dqn"]["policy_kwargs"]:
        raise ValueError("DQN policy weights require the same network architecture")
    path = Path(source["path"]).resolve(strict=True)
    if source["kind"] == "weights":
        initial = DoubleDQN.load(path, device=device)
        source_steps = initial.num_timesteps
        model = DoubleDQN(policy, env, seed=seed, device=device, **parameters)
        model.policy.load_state_dict(initial.policy.state_dict(), strict=True)
        model.exploration_decay_steps = exploration_steps
    else:
        replay = path.with_suffix(".replay.pkl").resolve(strict=True)
        manifest = json.loads(path.with_suffix(".replay.json").read_text(encoding="utf-8"))
        if manifest["model_sha256"] != file_hash(path) or manifest["replay_sha256"] != file_hash(replay):
            raise ValueError("DQN model/replay checkpoint integrity mismatch")
        model = DoubleDQN.load(path, env=env, device=device)
        model.load_replay_buffer(replay)
        if model.replay_buffer.n_envs != env.num_envs:
            raise ValueError("continued DQN must retain the number of environments")
        if hasattr(model.replay_buffer, "cut_trajectories"):
            model.replay_buffer.cut_trajectories()
        model.set_random_seed(seed)
        source_steps = model.num_timesteps
    evidence = {"sha256": file_hash(path), "source_steps": source_steps}
    if source["kind"] == "checkpoint":
        evidence["replay_sha256"] = manifest["replay_sha256"]
    return model, source | evidence

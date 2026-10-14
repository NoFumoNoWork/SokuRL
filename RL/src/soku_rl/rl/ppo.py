"""The single PPO implementation and checkpoint lifecycle for every training method."""
import hashlib
from pathlib import Path

from gymnasium import spaces
from hydra.utils import get_class
from stable_baselines3 import PPO

from soku_rl.policy.contract import read_training_contract
from soku_rl.rl import validate_payoff
from soku_rl.rl.action_initialization import initialize_action_bias, logical_action_prior


def algorithm_type(policy_type):
    if policy_type == "mlp":
        return PPO
    if policy_type == "lstm":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    raise ValueError("PPO policy_type must be mlp or lstm")


def create_ppo(env, interface, config, source, device, seed):
    algorithm = algorithm_type(config["policy_type"])
    policy = ("MultiInput" if isinstance(env.observation_space, spaces.Dict) else
              "Cnn" if len(env.observation_space.shape) == 3 else "Mlp")
    policy += "LstmPolicy" if config["policy_type"] == "lstm" else "Policy"
    model, metadata = initialize_ppo(algorithm, policy, env, interface, config, source, device, seed)
    if config["ppo"].get("freeze_actor_representation", False):
        from soku_rl.rl.frozen_representation import freeze_actor_representation
        freeze_actor_representation(model)
    if "rehearsal" in config:
        from soku_rl.rl.rehearsal import attach_rehearsal
        attach_rehearsal(model, interface, config["rehearsal"], source["kind"] == "checkpoint")
    if "online_anchor" in config:
        from soku_rl.rl.online_anchor import attach_anchor
        attach_anchor(model, interface, config["online_anchor"], source["kind"] == "checkpoint")
    if "online_teacher" in config:
        from soku_rl.rl.online_teacher import attach_teacher
        attach_teacher(model, interface, config["online_teacher"], source["kind"] == "checkpoint")
    return model, metadata


def snapshot(name, model, path):
    from soku_rl.policy.population import PPOPolicy
    from soku_rl.policy.recurrent import RecurrentPPOPolicy
    model.save(path)
    policy = PPOPolicy if isinstance(model, PPO) else RecurrentPPOPolicy
    return policy(name, model, Path(path))


def parameter_hash(policy):
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_ppo(algorithm, policy_type, env, interface, config, source, device, seed):
    from soku_rl.rl.recurrent_storage import attach_requested_storage
    validate_payoff(interface, config)
    if sum(option in config for option in ("online_anchor", "rehearsal", "online_teacher")) > 1:
        raise ValueError("select online_anchor, rehearsal or online_teacher; combining auxiliary objectives is not supported")
    parameters = dict(config["ppo"])
    if "freeze_actor_representation" in parameters:
        if type(parameters.pop("freeze_actor_representation")) is not bool:
            raise ValueError("freeze_actor_representation must be a boolean")
    if "recurrent_storage" in parameters:
        from sb3_contrib import RecurrentPPO
        storage = parameters.pop("recurrent_storage")
        if (algorithm is not RecurrentPPO or storage != "sparse"
                or "rollout_buffer_class" in parameters):
            raise ValueError("recurrent_storage requires recurrent PPO, sparse mode and no other buffer override")
    if "action_factorization" in parameters:
        from sb3_contrib import RecurrentPPO
        from soku_rl.rl.factorized_policy import FactorizedActorCriticPolicy
        factorization = parameters.pop("action_factorization")
        if (algorithm not in (PPO, RecurrentPPO) or interface.commands != tuple(range(576))
                or set(factorization) != {"button_probability"}
                or "action_persistence" in parameters or "initial_action_prior" in parameters):
            raise ValueError("factorized actions require full-command PPO without another action-head option")
        policy_type = FactorizedActorCriticPolicy
        if algorithm is RecurrentPPO:
            from soku_rl.rl.recurrent_factorized_policy import FactorizedRecurrentActorCriticPolicy
            policy_type = FactorizedRecurrentActorCriticPolicy
    if "action_persistence" in parameters:
        from sb3_contrib import RecurrentPPO
        from soku_rl.rl.persistent_policy import PersistentActorCriticPolicy
        persistence = parameters.pop("action_persistence")
        if (algorithm not in (PPO, RecurrentPPO) or interface.commands != tuple(range(576))
                or interface.config.action_history < 1
                or set(persistence) != {"repeat_probability"}):
            raise ValueError("action persistence requires PPO, full commands and action history")
        policy_type = PersistentActorCriticPolicy
        if algorithm is RecurrentPPO:
            from soku_rl.rl.recurrent_persistent_policy import PersistentRecurrentActorCriticPolicy
            policy_type = PersistentRecurrentActorCriticPolicy
    if "initial_action_prior" in parameters:
        prior_logits = logical_action_prior(interface, parameters.pop("initial_action_prior"))
    if "action_frame" in parameters:
        from sb3_contrib import RecurrentPPO
        from soku_rl.rl.facing_policy import FacingRecurrentActorCriticPolicy
        if (parameters.pop("action_frame") != "own_facing" or algorithm is not RecurrentPPO
                or interface.commands != tuple(range(576))
                or interface.episode.observation_mode != "privileged_state"
                or any(key in config["ppo"] for key in (
                    "action_factorization", "action_persistence", "initial_action_prior"))):
            raise ValueError("facing actions require full privileged recurrent PPO without another action-head option")
        policy_type = FacingRecurrentActorCriticPolicy
    architecture = dict(parameters["policy_kwargs"])
    if "action_frame" in config["ppo"]:
        from soku_rl.env.observation.memory_schema import FIGHTER_NAMES, PRIVILEGED_FEATURES, WORLD_NAMES
        if "facing_index" in architecture:
            raise ValueError("facing_index is derived from the shared observation contract")
        architecture["facing_index"] = ((interface.episode.history_frames - 1) * PRIVILEGED_FEATURES
            + 2 * (len(WORLD_NAMES) + FIGHTER_NAMES.index("dir")))
    if "action_factorization" in config["ppo"]:
        architecture["factor_button_probability"] = factorization["button_probability"]
    if "action_persistence" in config["ppo"]:
        architecture["repeat_probability"] = persistence["repeat_probability"]
    if "features_extractor_class" in architecture:
        architecture["features_extractor_class"] = get_class(architecture["features_extractor_class"])
        from soku_rl.rl.canonical_features import CanonicalCombatFeatures
        if (issubclass(architecture["features_extractor_class"], CanonicalCombatFeatures)
                and config["ppo"].get("action_frame") != "own_facing"):
            raise ValueError("canonical features require the own_facing action head")
    parameters["policy_kwargs"] = architecture
    if "rollout_buffer_class" in parameters:
        parameters["rollout_buffer_class"] = get_class(parameters["rollout_buffer_class"])
    elif algorithm is PPO and interface.episode.observation_mode == "privileged_state":
        from soku_rl.rl.buffers import PackedRolloutBuffer
        parameters["rollout_buffer_class"] = PackedRolloutBuffer
    if "rehearsal" in config:
        from soku_rl.rl.rehearsal import rehearsal_algorithm, validate_rehearsal
        validate_rehearsal(config["rehearsal"])
        algorithm = rehearsal_algorithm(algorithm)
    if "online_anchor" in config:
        from soku_rl.rl.online_anchor import anchored_algorithm, validate_anchor
        validate_anchor(config["online_anchor"])
        algorithm = anchored_algorithm(algorithm)
    if "online_teacher" in config:
        from soku_rl.rl.online_teacher import teacher_algorithm, validate_teacher
        validate_teacher(config["online_teacher"])
        algorithm = teacher_algorithm(algorithm)
    if source == {"kind": "fresh"}:
        model = algorithm(policy_type, env, seed=seed, device=device, **parameters)
        if "initial_action_prior" in config["ppo"]:
            initialize_action_bias(model, prior_logits)
        return attach_requested_storage(model, config), source
    if set(source) != {"kind", "path", "training_config"} or source["kind"] not in {"checkpoint", "weights"}:
        raise ValueError("initial policy must be fresh, a training checkpoint, or policy weights")
    training = read_training_contract(source["training_config"], interface)
    previous = training["rl"] if "rl" in training else training["algorithm"]
    if any(previous[key] != config[key] for key in ("policy_type", "timeout_payoff")):
        raise ValueError("PPO initialization requires the same policy type and payoff")
    if source["kind"] == "checkpoint" and previous["ppo"] != config["ppo"]:
        raise ValueError("continued PPO must retain its algorithm and optimizer configuration")
    if source["kind"] == "checkpoint":
        for option in ("rehearsal", "online_anchor", "online_teacher"):
            if ({key: previous[key] for key in (option,) if key in previous}
                    != {key: config[key] for key in (option,) if key in config}):
                raise ValueError(f"continued PPO must retain its {option} configuration")
    path = Path(source["path"]).resolve(strict=True)
    if source["kind"] == "weights":
        heads = ("action_persistence", "action_factorization", "action_frame")
        if (previous["ppo"]["policy_kwargs"] != config["ppo"]["policy_kwargs"]
                or {key: previous["ppo"][key] for key in heads if key in previous["ppo"]}
                != {key: config["ppo"][key] for key in heads if key in config["ppo"]}):
            raise ValueError("policy weights require the same network architecture")
        initial = algorithm.load(path, device=device)
        source_steps = initial.num_timesteps
        model = algorithm(policy_type, env, seed=seed, device=device, **parameters)
        model.policy.load_state_dict(initial.policy.state_dict(), strict=True)
    else:
        model = algorithm.load(path, env=env, device=device)
        model.set_random_seed(seed)
        source_steps = model.num_timesteps
    return attach_requested_storage(model, config), source | {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "source_steps": source_steps}

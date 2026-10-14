"""Evaluate a saved BR or rule reference with paired seats and one fixed character."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf


def candidate_specification(config, source, algorithm):
    candidate = config["candidate"]
    if candidate.get("kind") == "teacher_takeover":
        from soku_rl.policy.takeover import validate_takeover_frame
        if (set(candidate) != {"kind", "teacher", "after_frames"}
                or not isinstance(candidate["teacher"], dict)
                or set(candidate["teacher"]) != {"kind", "name", "rules"}
                or candidate["teacher"]["kind"] != "rule"
                or candidate["teacher"]["name"] != "god"):
            raise ValueError("teacher_takeover diagnostic requires an explicit God teacher")
        validate_takeover_frame(candidate["after_frames"])
        _, learner, metadata = candidate_specification(config | {"candidate": {"kind": "checkpoint"}},
                                                     source, algorithm)
        return "diagnostic-br:teacher-takeover", dict(candidate) | {"learner": learner}, metadata
    if set(candidate) == {"kind", "path"} and candidate["kind"] == "onnx":
        manifest = Path(candidate["path"]).resolve(strict=True)
        return "learned-br:onnx", {"kind": "onnx", "path": str(manifest)}, {
            "deployment_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    greedy = candidate == {"kind": "checkpoint", "inference": "greedy"}
    if candidate == {"kind": "checkpoint"} or greedy:
        model = (source / config["checkpoint"]).resolve(strict=True)
        if not model.is_relative_to(source):
            raise ValueError("checkpoint must belong to training_directory")
        from soku_rl.rl.learner import artifact_kind
        kind = artifact_kind(algorithm)
        spec = {"kind": kind, "path": str(model), "training_config": str(source / "config.yaml")}
        metadata = {"checkpoint_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
        if greedy and kind != "sb3_dqn":
            return "learned-br:greedy", {"kind": "greedy", "policy": spec}, metadata
        return "learned-br", spec, metadata
    if set(candidate) == {"kind", "name", "rules"} and candidate["kind"] == "rule":
        if not isinstance(candidate["name"], str) or not candidate["name"]:
            raise ValueError("rule reference requires a rule name")
        return f"rule-br:{candidate['name']}", dict(candidate), {}
    raise ValueError("BR candidate must be a checkpoint, ONNX deployment, rule reference or teacher_takeover")


@hydra.main(version_base="1.3", config_path="../config", config_name="benchmark_br")
def main(cfg):
    import torch
    from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
    from soku_rl.env.worker_pipe import WorkerBackend
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface, LearningVectorEnv
    from soku_rl.evaluation.br import benchmark_br
    from soku_rl.policy.loader import load_policy
    from soku_rl.policy.batch import evaluation_inference
    from soku_rl.policy.matchups import opponent_interface, select_opponents
    from soku_rl.policy.population import SeatPolicies
    from soku_rl.rl import configure_runtime

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    _, inference_name = evaluation_inference(config["benchmark"])
    configure_runtime(config["rl"])
    source = Path(config["training_directory"]).resolve(strict=True)
    training_path = source / "config.yaml"
    training = OmegaConf.to_container(OmegaConf.load(training_path), resolve=True, throw_on_missing=True)
    algorithm = training["algorithm"]
    if algorithm["name"] != "br" or algorithm["matchups"]["mode"] != "sampled":
        raise ValueError("cross-seat evaluation requires a sampled-matchup BR artifact")
    if config["require_complete"]:
        result = json.loads((source / "result.json").read_text(encoding="utf-8"))
        if result["success"] is not True or result["algorithm"] != "br":
            raise ValueError("completed BR evaluation requires successful training")
    if config["opponent_source"] == "training":
        population = algorithm["opponents"]
    elif config["opponent_source"] == "config":
        population = config["algorithm"]["opponents"]
    else:
        raise ValueError("opponent_source must be training or config")
    population = select_opponents(population, config["opponent_names"])
    candidate, specification, candidate_metadata = candidate_specification(config, source, algorithm)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
    episode = EpisodeConfig.from_dict(training["episode"])
    wrappers = LearningConfig(**training["wrappers"])
    interface = LearningInterface(episode, wrappers)
    learner = algorithm["matchups"]["learner"]
    if candidate in {p["name"] for p in population}:
        raise ValueError("candidate name collides with an opponent")
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Persist the actual source contract and opponent setups, not train defaults.
    config.update(source_training=training, evaluation_opponents=population,
        policy_inference=inference_name,
        evaluation_candidate={"name": candidate, "policy": specification}, **candidate_metadata,
        training_config_sha256=hashlib.sha256(training_path.read_bytes()).hexdigest())
    (output / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)), encoding="utf-8")
    report = {"success": False, "phase": "loading_policies",
        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    started = time.perf_counter()
    try:
        policy = load_policy(candidate, specification, interface, device)
        report["candidate_fingerprint"] = policy.fingerprint
        strategies = {candidate: SeatPolicies(candidate, (policy, policy))}
        setups = {}
        for entry in population:
            matched = opponent_interface(interface, learner, entry)
            opponent = load_policy(entry["name"], entry["policy"], matched, device)
            strategies[entry["name"]] = SeatPolicies(entry["name"], (opponent, opponent))
            setups[entry["name"]] = entry["setup"]
        # SB3 seed initialization can change cuDNN flags while loading a model.
        # Record effective inference settings after every policy has loaded.
        report["inference_numerics"] = {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
        report["phase"] = "running_games"
        with closing(WorkerBackend(log_path=output / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(episode.backend_observation())
            report["runtime"] = backend.identity
            env = LearningVectorEnv(TwoPlayerVectorEnv(backend, config["num_envs"], episode), wrappers)
            report["result"] = benchmark_br(env, strategies, candidate, learner, setups,
                config["benchmark"], backend.identity["fingerprints"]["game_id"], output)
        report["success"] = True
        report["phase"] = "finished"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Supervised initialization of shared PPO actors or DQN online action values."""
import hashlib
import json
import math
from pathlib import Path
import time

from gymnasium import Env
import numpy as np
import torch

from soku_rl.policy.contract import read_training_contract
from soku_rl.rl.learner import create_learner, learner_kind, parameter_hash
from soku_rl.rl.dqn import DoubleDQN
from soku_rl.rl.sparse_transfer import restore_batch
from soku_rl.rl.storage import PackedObservation
from soku_rl.rl.command_diagnostics import command_group_totals, summarize_command_groups
from soku_rl.rl.demonstration_loss import validate_change_weight, weighted_action_loss, weighted_validation_nll
from soku_rl.rl.demonstration_supervision import supervised_samples
from soku_rl.rl.noisy_demonstrations import validate_noise_manifest, validate_noise_episode


class ObservationContractEnv(Env):
    """Supply spaces to the PPO factory; offline fitting must never simulate steps."""
    def __init__(self, interface):
        self.observation_space = interface.observation_space
        self.action_space = interface.action_space

    def reset(self, *, seed, options):
        raise RuntimeError("offline initialization cannot reset a game")

    def step(self, action):
        raise RuntimeError("offline initialization cannot advance a game")


def load_demonstrations(source, interface):
    source = Path(source).resolve(strict=True)
    contract = read_training_contract(source / "config.yaml", interface)
    report = json.loads((source / "result.json").read_text())
    path = source / "episodes/manifest.json"
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (report["success"] is not True or report["method"] != "rule_demonstrations"
            or manifest["complete"] is not True or manifest["schema"] not in (1, 2, 3, 4)
            or manifest != report["result"] or manifest["incomplete_episodes"]):
        raise ValueError("demonstrations require a successful complete collection and matching manifest")
    if manifest["schema"] == 2 and (manifest["control"] not in {"teacher", "learner"}
            or not isinstance(manifest["behavior_fingerprint"], str) or not manifest["behavior_fingerprint"]
            or (manifest["control"] == "teacher"
                and manifest["behavior_fingerprint"] != manifest["teacher_fingerprint"])):
        raise ValueError("invalid demonstration behavior identity")
    if manifest["schema"] == 3:
        after = manifest["after_frames"]
        if (manifest["control"] != "teacher_takeover" or type(after) is not int or after < 0
                or not isinstance(manifest["learner_fingerprint"], str) or not manifest["learner_fingerprint"]
                or manifest["supervision"] != "executed_teacher_suffix_with_complete_unsupervised_prefix"):
            raise ValueError("invalid recovery demonstration identity or supervision")
        identity = ["teacher-takeover-v1", manifest["learner_fingerprint"], manifest["teacher_fingerprint"], after]
        if hashlib.sha256(json.dumps(identity).encode()).hexdigest() != manifest["behavior_fingerprint"]:
            raise ValueError("recovery demonstration composite fingerprint differs")
        behavior = contract["behavior"]
        if (behavior["kind"] != "teacher_takeover" or behavior["after_frames"] != after
                or behavior["teacher"] != contract["teacher"]):
            raise ValueError("recovery demonstration behavior contract differs")
    if manifest["schema"] == 4:
        validate_noise_manifest(manifest, contract)
    if (manifest["observation_shape"] != list(interface.observation_space.shape)
            or manifest["observation_dtype"] != interface.observation_space.dtype.str
            or manifest["num_actions"] != interface.action_space.n):
        raise ValueError("demonstration observation/action contract differs")
    episodes = manifest["episodes"]
    if (not episodes or len({e["world_seed"] for e in episodes}) != len(episodes)
            or len({e["id"] for e in episodes}) != len(episodes)
            or len({e["path"] for e in episodes}) != len(episodes)):
        raise ValueError("demonstration episodes must have unique identities and world seeds")
    excluded = {seed for block in contract["excluded"].values() for seed in block["world_seeds"]}
    if excluded & {e["world_seed"] for e in episodes}:
        raise ValueError("demonstration worlds overlap reserved evaluation seeds")
    plan = {row["id"]: row for row in json.loads((source / "episodes/plan.json").read_text())}
    if set(plan) != {row["id"] for row in episodes}:
        raise ValueError("demonstration manifest does not complete its plan")
    samples = {"train": [], "validation": []}
    seats = {"train": set(), "validation": set()}
    for row in episodes:
        if row["split"] not in samples or any(row[key] != value for key, value in plan[row["id"]].items()):
            raise ValueError("demonstration split or identity differs from its plan")
        episode_path = (source / "episodes" / row["path"]).resolve(strict=True)
        if not episode_path.is_relative_to(source / "episodes"):
            raise ValueError("demonstration shard must belong to its dataset")
        with episode_path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise ValueError("demonstration shard hash differs")
        # These are trusted local shards, verified before decoding custom packed arrays.
        data = torch.load(episode_path, map_location="cpu", weights_only=False)
        count = row["steps"]
        if (data["schema"] != manifest["schema"] or count < 1 or len(data["observations"]) != count
                or any(data[key].shape != (count,) for key in ("actions", "rewards", "returns"))
                or data["actions"].dtype != np.int64 or (data["actions"] < 0).any()
                or (data["actions"] >= interface.action_space.n).any()
                or not np.isfinite(data["rewards"]).all() or not np.isfinite(data["returns"]).all()):
            raise ValueError("invalid demonstration sample arrays")
        executed = data["actions"]
        if data["schema"] >= 2:
            executed = data["executed_actions"]
            if (executed.shape != (count,) or executed.dtype != np.int64 or (executed < 0).any()
                    or (executed >= interface.action_space.n).any()
                    or np.count_nonzero(executed != data["actions"]) != row["teacher_behavior_disagreements"]
                    or (manifest["control"] == "teacher" and not np.array_equal(executed, data["actions"]))):
                raise ValueError("invalid demonstration executed actions or teacher agreement")
        if data["schema"] == 3:
            mask = data["supervised"]
            if (mask.dtype != np.bool_ or mask.shape != (count,)
                    or not np.array_equal(mask, np.arange(count) >= manifest["after_frames"])
                    or type(row["supervised_steps"]) is not int or int(mask.sum()) != row["supervised_steps"]
                    or not np.array_equal(executed[mask], data["actions"][mask])):
                raise ValueError("invalid recovery supervision mask or executed teacher suffix")
        if data["schema"] == 4:
            validate_noise_episode(data, row, manifest)
        expected = np.cumsum(data["rewards"][::-1], dtype=np.float64)[::-1].astype(np.float32)
        if not np.array_equal(expected, data["returns"]):
            raise ValueError("demonstration return targets do not match episode rewards")
        for observation in data["observations"]:
            if (not isinstance(observation, PackedObservation) or observation.shape != interface.observation_space.shape
                    or observation.dtype != interface.observation_space.dtype.str):
                raise ValueError("invalid packed demonstration observation")
        # -1 marks the first frame: never count a transition across episode boundaries.
        # On learner trajectories the copy baseline uses the actual preceding input,
        # not an unexecuted teacher label that would be unavailable to the learner.
        changes = np.concatenate(([-1], np.not_equal(data["actions"][1:], executed[:-1]).astype(int)))
        arrays = (data["observations"], data["actions"], data["returns"], changes)
        if data["schema"] == 3:
            arrays += (data["supervised"],)
        samples[row["split"]].extend(zip(*arrays, strict=True))
        seats[row["split"]].add(row["learner_seat"])
    if any(value != {0, 1} for value in seats.values()):
        raise ValueError("both demonstration splits must cover both learner seats")
    if sum(len(rows) for rows in samples.values()) != manifest["successful_env_steps"]:
        raise ValueError("demonstration step accounting differs")
    return samples, manifest, contract, hashlib.sha256(manifest_bytes).hexdigest()


def supervised_predictions(model, observations, actions):
    if isinstance(model, DoubleDQN):
        from stable_baselines3.common.distributions import CategoricalDistribution
        q_values = model.q_net(observations)
        # Softmax is a supervised ranking loss, never DQN's deployed policy.
        distribution = CategoricalDistribution(model.action_space.n).proba_distribution(q_values)
        return distribution, q_values.gather(1, actions[:, None]).flatten()
    return model.policy.get_distribution(observations), model.policy.predict_values(observations).flatten()


def score_samples(model, samples, batch_size):
    samples = supervised_samples(samples)
    if not samples:
        raise ValueError("scoring requires supervised demonstration frames")
    model.policy.set_training_mode(False)
    totals = dict(nll=0., accuracy=0., value_mse=0., entropy=0.)
    changed_count, changed_correct, changed_nll = 0, 0., 0.
    command_totals = {}
    with torch.no_grad():
        for first in range(0, len(samples), batch_size):
            batch = samples[first:first + batch_size]
            observations, actions, returns = sample_tensors(model, batch)
            distribution, values = supervised_predictions(model, observations, actions)
            nll = -distribution.log_prob(actions)
            correct = distribution.mode() == actions
            totals["nll"] += float(nll.sum())
            totals["accuracy"] += float(correct.sum())
            totals["value_mse"] += float(((values - returns) ** 2).sum())
            totals["entropy"] += float(distribution.entropy().sum())
            changed = torch.as_tensor([row[3] == 1 for row in batch], device=model.device)
            changed_count += int(changed.sum())
            changed_correct += float(correct[changed].sum())
            changed_nll += float(nll[changed].sum())
            if model.action_space.n == 576:
                diagnostics = command_group_totals(distribution, actions, torch.ones_like(actions, dtype=torch.bool))
                for key, value in diagnostics.items():
                    command_totals[key] = command_totals.get(key, 0) + value
    result = {key: value / len(samples) for key, value in totals.items()}
    result["changed_samples"] = changed_count
    if changed_count:
        result.update(changed_accuracy=changed_correct / changed_count, changed_nll=changed_nll / changed_count)
    if command_totals:
        result.update(summarize_command_groups(command_totals))
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite behavior-cloning validation metrics")
    return result


def sample_tensors(model, samples):
    observations = restore_batch([row[0] for row in samples], model.device)
    return observations, torch.as_tensor(np.asarray([row[1] for row in samples]), device=model.device), \
        torch.as_tensor(np.asarray([row[2] for row in samples], dtype=np.float32), device=model.device)


def fit_demonstrations(interface, algorithm, samples, config, device, seed, directory):
    kind = learner_kind(algorithm)
    for key in ("epochs", "batch_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"pretraining {key} must be a positive integer")
    if (type(config["value_coef"]) not in (int, float) or not math.isfinite(config["value_coef"])
            or config["value_coef"] < 0 or algorithm["policy_type"] not in {"mlp", "lstm"}
            or algorithm[kind]["gamma"] != 1. or any(not rows for rows in samples.values())):
        raise ValueError("pretraining requires nonempty numeric samples, a shared learner and gamma=1")
    recurrent = algorithm["policy_type"] == "lstm"
    labelled = {split: supervised_samples(rows) for split, rows in samples.items()}
    if any(not rows for rows in labelled.values()):
        raise ValueError("each demonstration split requires supervised frames")
    if config["value_coef"] and any(len(labelled[split]) != len(rows) for split, rows in samples.items()):
        raise ValueError("masked recovery demonstrations require value_coef=0")
    validate_change_weight(config["action_change_weight"])
    if recurrent:
        from soku_rl.rl.recurrent_cloning import demonstration_episodes, sequence_epoch
        if ("sequence_length" not in config or type(config["sequence_length"]) is not int
                or config["sequence_length"] < 1 or config["batch_size"] % config["sequence_length"]):
            raise ValueError("recurrent batch_size must be a positive multiple of sequence_length")
        episodes = {split: demonstration_episodes(rows) for split, rows in samples.items()}
    elif "sequence_length" in config:
        raise ValueError("sequence_length requires a recurrent policy")
    view = ObservationContractEnv(interface)
    if config["initial_policy"]["kind"] not in {"fresh", "weights"}:
        raise ValueError("supervised initialization requires fresh or weights with a fresh optimizer")
    model, source = create_learner(view, interface, algorithm, config["initial_policy"], device, seed)
    initial = parameter_hash(model.policy)
    rng = np.random.default_rng(seed)
    label_counts = {split: np.bincount([int(row[1]) for row in rows],
        minlength=interface.action_space.n) for split, rows in labelled.items()}
    majority = int(label_counts["train"].argmax())
    baseline = {"train_majority_action": majority,
        "train_majority_fraction": float(label_counts["train"][majority] / len(labelled["train"])),
        "validation_accuracy": float(label_counts["validation"][majority] / len(labelled["validation"])),
        "label_counts": {split: counts.tolist() for split, counts in label_counts.items()}}
    repeat_baselines = {}
    for split, rows in labelled.items():
        transitions = sum(row[3] >= 0 for row in rows)
        repeat_baselines[split] = {"transitions": int(transitions)}
        if transitions:
            repeat_baselines[split]["accuracy"] = float(sum(row[3] == 0 for row in rows) / transitions)
    def validation_score():
        if recurrent:
            metrics = sequence_epoch(model, episodes["validation"], np.arange(len(episodes["validation"])),
                config["batch_size"], config["sequence_length"], config["value_coef"], False,
                config["action_change_weight"])[0]
        else:
            metrics = score_samples(model, samples["validation"], config["batch_size"])
        metrics["weighted_nll"] = weighted_validation_nll(metrics, len(labelled["validation"]), config["action_change_weight"])
        return metrics

    validation = validation_score()
    history = [{"epoch": 0, "validation": validation}]
    best, best_epoch = validation["weighted_nll"], 0
    model.save(directory / "initial.zip")
    model.save(directory / "best.zip")
    started = time.perf_counter()
    updates = 0
    for epoch in range(1, config["epochs"] + 1):
        if recurrent:
            _, train_loss, epoch_updates = sequence_epoch(model, episodes["train"],
                rng.permutation(len(episodes["train"])), config["batch_size"], config["sequence_length"],
                config["value_coef"], True, config["action_change_weight"])
            updates += epoch_updates
        else:
            model.policy.set_training_mode(True)
            total_loss = 0.
            order = rng.permutation(len(labelled["train"]))
            for first in range(0, len(order), config["batch_size"]):
                batch = [labelled["train"][int(i)] for i in order[first:first + config["batch_size"]]]
                observations, actions, returns = sample_tensors(model, batch)
                if isinstance(model, DoubleDQN):
                    distribution, values = supervised_predictions(model, observations, actions)
                    log_probs = distribution.log_prob(actions)
                else:
                    values, log_probs, _ = model.policy.evaluate_actions(observations, actions)
                changed = torch.as_tensor([row[3] == 1 for row in batch], device=model.device)
                loss = weighted_action_loss(-log_probs, changed, torch.ones_like(changed), config["action_change_weight"])
                loss = loss + config["value_coef"] * ((values.flatten() - returns) ** 2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite behavior-cloning loss")
                model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                model.policy.optimizer.step()
                updates += 1
                total_loss += float(loss.detach()) * len(batch)
            train_loss = total_loss / len(order)
        if isinstance(model, DoubleDQN):
            model.q_net_target.load_state_dict(model.q_net.state_dict())
        validation = validation_score()
        if validation["weighted_nll"] < best:
            best, best_epoch = validation["weighted_nll"], epoch
            model.save(directory / "best.zip")
        row = {"epoch": epoch, "train_loss": train_loss, "validation": validation,
            "updates": updates, "seconds": time.perf_counter() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        (directory / "progress.json").write_text(json.dumps({"method": "behavior_cloning",
            "epochs": history, "best_epoch": best_epoch}, indent=2), encoding="utf-8")
    final = parameter_hash(model.policy)
    if initial == final:
        raise RuntimeError("behavior cloning did not update the policy")
    model.save(directory / "final.zip")
    return {"checkpoint": str(directory / "best.zip"), "final_checkpoint": str(directory / "final.zip"),
        "learner": kind, "learner_steps": model.num_timesteps,
        **({"ppo_steps": model.num_timesteps} if kind == "ppo" else
           {"supervision": "softmax_q_action_ranking_and_teacher_action_return", "inference": "greedy_online_q"}),
        "supervised_updates": updates, "best_epoch": best_epoch,
        "initialization": source,
        "selection": {"metric": "weighted_nll", "action_change_weight": config["action_change_weight"]},
        "initial_policy_hash": initial, "final_policy_hash": final,
        "constant_action_baseline": baseline,
        "copy_previous_action_baseline": repeat_baselines,
        **({"sequence_training": {"sequence_length": config["sequence_length"],
            "episodes_per_batch": config["batch_size"] // config["sequence_length"],
            "state_reset": "each_episode", "gradient_truncation": "each_chunk"}} if recurrent else {}),
        "train_frames": len(samples["train"]), "validation_frames": len(samples["validation"]),
        "supervised_frames": {split: len(rows) for split, rows in labelled.items()}, "history": history}

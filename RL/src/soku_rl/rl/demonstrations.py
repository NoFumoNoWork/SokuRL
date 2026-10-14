"""Collect complete expert episodes with disjoint training and validation worlds."""
from dataclasses import asdict
import hashlib
import json
import time

import numpy as np
import torch

from soku_rl.env.encoding import AGENTS
from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.rl.storage import PackedObservation
from soku_rl.policy.takeover import TeacherTakeoverPolicy
from soku_rl.policy.action_noise import ActionNoisePolicy


def demonstration_plan(config, population, seed, excluded_worlds):
    per_seat, held_out = config["episodes_per_seat"], config["validation_per_seat"]
    if (type(per_seat) is not int or type(held_out) is not int
            or not 0 < held_out < per_seat):
        raise ValueError("demonstrations need training and validation episodes in both seats")
    names = [entry["name"] for entry in population]
    weights = np.asarray([entry["probability"] for entry in population], dtype=float)
    if (not names or len(set(names)) != len(names) or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1.)):
        raise ValueError("demonstration opponents require a named probability distribution")
    rng = np.random.default_rng(seed)
    used = set(excluded_worlds)
    plan = []
    for seat in (0, 1):
        for episode in range(per_seat):
            world = int(rng.integers(0, 0xFFFFFFFF))
            while world in used:
                world = int(rng.integers(0, 0xFFFFFFFF))
            used.add(world)
            index = int(rng.choice(len(population), p=weights))
            plan.append({"id": len(plan), "split": "validation" if episode < held_out else "train",
                "learner_seat": seat, "opponent_index": index, "opponent": names[index],
                "world_seed": world, "teacher_seed": int(rng.integers(0, 0xFFFFFFFF)),
                "opponent_seed": int(rng.integers(0, 0xFFFFFFFF)),
                "behavior_seed": int(rng.integers(0, 0xFFFFFFFF))})
    return [plan[int(index)] for index in rng.permutation(len(plan))]


def collect_demonstrations(env, plan, learner, population, teacher, opponents, behavior, directory):
    takeover = isinstance(behavior, TeacherTakeoverPolicy)
    noisy_teacher = isinstance(behavior, ActionNoisePolicy)
    if takeover and (behavior.teacher.fingerprint != teacher.fingerprint
            or env.interface.episode.decision_frames != 1 or env.interface.episode.latency_frames != 0):
        raise ValueError("recovery collection requires the declared teacher and every-frame zero-latency control")
    if noisy_teacher and (behavior.policy.fingerprint != teacher.fingerprint
            or behavior.num_actions != env.single_action_space.n
            or env.interface.episode.decision_frames != 1 or env.interface.episode.latency_frames != 0):
        raise ValueError("noisy collection requires the declared teacher and every-frame zero-latency control")
    if env.single_observation_space.shape is None or len(env.single_observation_space.shape) != 1:
        raise ValueError("demonstrations currently require flat numeric observations")
    if (not plan or len(opponents) != len(population)
            or len({job["id"] for job in plan}) != len(plan)
            or len({job["world_seed"] for job in plan}) != len(plan)
            or any(job["learner_seat"] not in (0, 1) or job["split"] not in {"train", "validation"}
                   or not 0 <= job["opponent_index"] < len(population) for job in plan)):
        raise ValueError("invalid or overlapping demonstration plan")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    metadata = {"schema": 4 if noisy_teacher else 3 if takeover else 2,
        "complete": False, "teacher_fingerprint": teacher.fingerprint,
        "control": "teacher_noise" if noisy_teacher else
            "teacher_takeover" if takeover else "teacher" if behavior is teacher else "learner",
        "behavior_fingerprint": behavior.fingerprint,
        "opponent_fingerprints": {entry["name"]: p.fingerprint
            for entry, p in zip(population, opponents, strict=True)},
        "observation_shape": list(env.single_observation_space.shape),
        "observation_dtype": env.single_observation_space.dtype.str,
        "num_actions": int(env.single_action_space.n), "episodes": [], "successful_env_steps": 0}
    if takeover:
        metadata.update(after_frames=behavior.after_frames,
            learner_fingerprint=behavior.learner.fingerprint,
            supervision="executed_teacher_suffix_with_complete_unsupervised_prefix")
    if noisy_teacher:
        metadata.update(random_probability=behavior.random_probability,
                        supervision="unperturbed_teacher_labels_all_frames")
    started = time.perf_counter()
    pending = {}
    try:
        for first in range(0, len(plan), env.num_envs):
            jobs = dict(enumerate(plan[first:first + env.num_envs]))
            matches, actors = {}, {}
            for slot, job in jobs.items():
                own = PlayerSetup(**learner)
                other = PlayerSetup(**population[job["opponent_index"]]["setup"])
                pair = (own, other) if job["learner_seat"] == 0 else (other, own)
                matches[slot] = MatchConfig(*pair)
                label_actor = (behavior.spawn_with_seeds(job["behavior_seed"], job["teacher_seed"])
                    if takeover or noisy_teacher else teacher.spawn(job["teacher_seed"]))
                actors[slot] = (label_actor,
                    opponents[job["opponent_index"]].spawn(job["opponent_seed"]),
                    label_actor if behavior is teacher or takeover or noisy_teacher else behavior.spawn(job["behavior_seed"]))
            observations, _ = env.reset_matchups({s: j["world_seed"] for s, j in jobs.items()}, matches)
            pending = {s: {"job": j, "observations": [], "actions": [], "executed_actions": [],
                "rewards": [], "noise_selected": []} for s, j in jobs.items()}
            while pending:
                actions = {}
                for slot, record in pending.items():
                    seat = record["job"]["learner_seat"]
                    own, other = AGENTS[seat], AGENTS[1 - seat]
                    if noisy_teacher:
                        label, executed, selected = actors[slot][0].act_with_label(observations[slot][own])
                        action, executed = int(label), int(executed)
                        record["noise_selected"].append(selected)
                    elif takeover:
                        label, executed = actors[slot][0].act_labelled(observations[slot][own])
                        action, executed = int(label), int(executed)
                    else:
                        action = int(actors[slot][0].act(observations[slot][own]))
                        executed = action if behavior is teacher else int(actors[slot][2].act(observations[slot][own]))
                    if not env.single_action_space.contains(action):
                        raise ValueError("teacher action is outside the learning vocabulary")
                    if not env.single_action_space.contains(executed):
                        raise ValueError("behavior action is outside the learning vocabulary")
                    record["observations"].append(PackedObservation.pack(observations[slot][own]))
                    record["actions"].append(action)
                    record["executed_actions"].append(executed)
                    actions[slot] = {own: executed,
                        other: actors[slot][1].act(observations[slot][other])}
                observations, rewards, terms, truncs, infos = env.step(actions)
                metadata["successful_env_steps"] += len(actions)
                for slot in list(pending):
                    record = pending[slot]
                    job = record["job"]
                    own = AGENTS[job["learner_seat"]]
                    record["rewards"].append(float(rewards[slot][own]))
                    if not (terms[slot][own] or truncs[slot][own]):
                        continue
                    info = infos[slot][own]
                    name = f"episode-{job['id']:06d}.pt"
                    reward_array = np.asarray(record["rewards"], dtype=np.float32)
                    # Undiscounted finite-horizon targets match the current shared PPO payoff.
                    returns = np.cumsum(reward_array[::-1], dtype=np.float64)[::-1].copy().astype(np.float32)
                    data = {"schema": metadata["schema"], "observations": record["observations"],
                        "actions": np.asarray(record["actions"], dtype=np.int64),
                        "executed_actions": np.asarray(record["executed_actions"], dtype=np.int64),
                        "rewards": reward_array, "returns": returns}
                    if takeover:
                        data["supervised"] = np.arange(len(reward_array)) >= behavior.after_frames
                    if noisy_teacher:
                        data["noise_selected"] = np.asarray(record["noise_selected"], dtype=bool)
                    torch.save(data, directory / name)
                    with (directory / name).open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    row = job | {"path": name, "sha256": digest, "steps": len(reward_array),
                        "match": asdict(matches[slot]), "outcome": info["outcome"], "frame": info["frame"],
                        "return": float(reward_array.sum(dtype=np.float64)),
                        "teacher_behavior_disagreements": int(np.count_nonzero(
                            np.asarray(record["actions"]) != np.asarray(record["executed_actions"])))}
                    if "combat_metrics" in info:
                        row["combat_metrics"] = info["combat_metrics"]
                    if takeover:
                        row["supervised_steps"] = int(data["supervised"].sum())
                    if noisy_teacher:
                        row["noise_decisions"] = int(data["noise_selected"].sum())
                    metadata["episodes"].append(row)
                    del pending[slot]
                    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata["complete"] = True
        return metadata
    finally:
        metadata["seconds"] = time.perf_counter() - started
        metadata["incomplete_episodes"] = [{"id": row["job"]["id"], "observed_steps": len(row["rewards"]),
            "attempted_actions": len(row["actions"])} for row in pending.values()]
        (directory / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

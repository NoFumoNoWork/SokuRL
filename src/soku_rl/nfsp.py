"""Train OpenSpiel NFSP with independently reset two-player game slots."""
import json
from pathlib import Path

import numpy as np
from .env.encoding import AGENTS
from .spiel_nfsp import VectorNFSP


def train_nfsp(env, config, device, seed, directory):
    if config["episodes"] < 1 or config["checkpoint_every"] < 1:
        raise ValueError("positive episode and checkpoint counts required")
    if config["timeout_payoff"] != "zero_at_horizon":
        raise ValueError("NFSP trainer requires explicit zero_at_horizon payoff")
    learner = VectorNFSP(env.single_observation_space.shape, config["agent"], device, seed)
    rng = np.random.default_rng(seed)
    count = min(env.num_envs, config["episodes"])
    seeds = {s: int(rng.integers(0, 0xFFFFFFFF)) for s in range(count)}
    observations, _ = env.reset(seeds)
    learner.begin(seeds)
    issued, finished, steps = count, 0, 0
    records = []
    destination = Path(directory)
    while observations:
        actions = learner.act(observations)
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        learner.feed(observations, actions, next_obs, rewards, terms, truncs)
        steps += len(actions)
        resets = {}
        for slot in list(next_obs):
            if terms[slot][AGENTS[0]] or truncs[slot][AGENTS[0]]:
                finished += 1
                records.append(infos[slot][AGENTS[0]] | {"slot": slot,
                                "terminal_rewards": rewards[slot],
                                "modes": {a: learner.modes[slot, a] for a in AGENTS}})
                del next_obs[slot]
                if issued < config["episodes"]:
                    resets[slot] = int(rng.integers(0, 0xFFFFFFFF))
                    issued += 1
                if finished % config["checkpoint_every"] == 0:
                    checkpoint = destination / f"episodes-{finished}"
                    checkpoint.mkdir()
                    learner.save(checkpoint)
        if resets:
            restarted, _ = env.reset(resets)
            learner.begin(resets)
            next_obs.update(restarted)
        observations = next_obs
        if resets or not observations:
            (destination / "progress.json").write_text(json.dumps({
                "episodes": finished, "environment_steps": steps, "games": records,
                "timeout_payoff": config["timeout_payoff"], "learning": learner.metrics()},
                indent=2), encoding="utf-8")
    checkpoint = destination / "final"
    checkpoint.mkdir()
    learner.save(checkpoint)
    if any(min(counts.values()) == 0 for counts in learner.updates.values()):
        raise RuntimeError("NFSP completed without both RL and SL updates for both players")
    return {"episodes": finished, "environment_steps": steps, "games": records,
            "learning": learner.metrics()}

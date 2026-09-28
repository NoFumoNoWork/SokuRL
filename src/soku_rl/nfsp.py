"""Collect simultaneous vector transitions for the public RLCard NFSP agent."""
import json
from pathlib import Path

import numpy as np
from rlcard.agents.nfsp_agent import NFSPAgent

from .env.encoding import AGENTS, NUM_ACTIONS


def rlcard_state(observation):
    # Every key combination is accepted by the game, including ineffective ones.
    return {"obs": observation, "legal_actions": dict.fromkeys(range(NUM_ACTIONS), None),
            "raw_legal_actions": list(range(NUM_ACTIONS))}


class VectorNFSP:
    """Two shared learners; one fixed best-response/average mode per lane/player.

    RLCard 1.2.0 exposes sampling through sample_episode_policy but stores the
    result in _mode. Only that field is switched across lanes. Replay buffers,
    optimizers, counters and networks remain shared within each player role.
    feed() accepts explicit transitions, so no cross-lane previous state exists.
    """
    def __init__(self, observation_shape, agent_config, device):
        self.agents = {a: NFSPAgent(num_actions=NUM_ACTIONS, state_shape=list(observation_shape),
                                   device=device, **agent_config) for a in AGENTS}
        self.modes = {}

    def begin(self, slots):
        for slot in slots:
            for name, agent in self.agents.items():
                agent.sample_episode_policy()
                self.modes[slot, name] = agent._mode

    def act(self, observations):
        actions = {}
        for slot, players in observations.items():
            actions[slot] = {}
            for name in AGENTS:
                agent = self.agents[name]
                agent._mode = self.modes[slot, name]
                actions[slot][name] = int(agent.step(rlcard_state(players[name])))
        return actions

    def feed(self, observations, actions, next_observations, rewards, terminated, truncated):
        for slot in actions:
            for name, agent in self.agents.items():
                # This reference trainer explicitly optimizes the finite-horizon
                # game with zero additional payoff at a time limit.
                done = terminated[slot][name] or truncated[slot][name]
                agent.feed((rlcard_state(observations[slot][name]), actions[slot][name],
                            rewards[slot][name], rlcard_state(next_observations[slot][name]), done))

    def save(self, directory):
        for name, agent in self.agents.items():
            agent.save_checkpoint(str(directory), name + ".pt")


def train_nfsp(env, config, device, seed, directory):
    if config["episodes"] < 1 or config["checkpoint_every"] < 1:
        raise ValueError("positive episode and checkpoint counts required")
    if config["timeout_payoff"] != "zero_at_horizon":
        raise ValueError("RLCard reference trainer requires explicit zero_at_horizon payoff")
    learner = VectorNFSP(env.single_observation_space.shape, config["agent"], device)
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
                "timeout_payoff": config["timeout_payoff"]}, indent=2), encoding="utf-8")
    checkpoint = destination / "final"
    checkpoint.mkdir()
    learner.save(checkpoint)
    return {"episodes": finished, "environment_steps": steps, "games": records}

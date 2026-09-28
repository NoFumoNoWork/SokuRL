"""Batch OpenSpiel NFSP inference and collect explicit per-game transitions."""
from pathlib import Path

import numpy as np
import torch
from open_spiel.python import rl_agent, rl_environment
from open_spiel.python.pytorch.nfsp import NFSP

from .env.encoding import AGENTS


def time_step(observations, rewards, done, legal_actions):
    return rl_environment.TimeStep(
        observations={"info_state": [observations[a] for a in AGENTS],
                      "legal_actions": [legal_actions, legal_actions], "current_player": -2},
        rewards=[rewards[a] for a in AGENTS], discounts=[0. if done else 1.] * 2,
        step_type=rl_environment.StepType.LAST if done else rl_environment.StepType.MID)


class VectorNFSP:
    """Share networks and buffers across slots, never previous-state pointers.

    Uses OpenSpiel 2.0.2 NFSP networks, reservoir sampling, supervised loss and
    DQN replay/loss. Collection is external: one mode per player per episode;
    every transition updates the common DQN clock, including average-policy play.
    """
    def __init__(self, observation_shape, num_actions, agent_config, device, seed):
        if len(observation_shape) != 1:
            raise ValueError("OpenSpiel NFSP currently requires a numeric state vector")
        self.config = dict(agent_config)
        if isinstance(num_actions, (bool, np.bool_)) or not isinstance(num_actions, (int, np.integer)) or num_actions < 1:
            raise ValueError("num_actions must be a positive integer")
        self.num_actions = int(num_actions)
        self.legal_actions = list(range(self.num_actions))
        self.shape = tuple(observation_shape)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.agents = {name: NFSP(index, observation_shape[0], num_actions,
                                 device=str(device), seed=seed + index, **agent_config)
                       for index, name in enumerate(AGENTS)}
        self.modes = {}
        self.updates = {name: {"rl": 0, "sl": 0} for name in AGENTS}

    def begin(self, slots):
        for slot in slots:
            for name in AGENTS:
                self.modes[slot, name] = ("best_response" if self.rng.random() <
                                          self.config["anticipatory_param"] else "average_policy")

    @torch.inference_mode()
    def act(self, observations):
        slots = tuple(observations)
        actions = {slot: {} for slot in slots}
        for name, agent in self.agents.items():
            states = torch.as_tensor(np.stack([observations[s][name] for s in slots]),
                                     device=self.device, dtype=torch.float32)
            q = agent._rl_agent
            greedy = q._q_network(states).argmax(-1).cpu().numpy()
            average = agent._avg_network(states).softmax(-1).cpu().numpy()
            epsilon = q.epsilon_schedule(q._iteration)
            for index, slot in enumerate(slots):
                if self.modes[slot, name] == "best_response":
                    action = int(self.rng.integers(self.num_actions) if self.rng.random() < epsilon
                                 else greedy[index])
                    # NFSP learns the empirical distribution of selected BR actions.
                    probabilities = np.zeros(self.num_actions, dtype=np.float32)
                    probabilities[action] = 1.
                    ts = time_step(observations[slot], dict.fromkeys(AGENTS, 0.), False, self.legal_actions)
                    agent.add_transition(ts, rl_agent.StepOutput(action, probabilities))
                else:
                    action = int(self.rng.choice(self.num_actions, p=average[index]))
                actions[slot][name] = action
        return actions

    def feed(self, observations, actions, next_observations, rewards, terminated, truncated):
        for slot, joint in actions.items():
            done = terminated[slot][AGENTS[0]] or truncated[slot][AGENTS[0]]
            previous = time_step(observations[slot], dict.fromkeys(AGENTS, 0.), False, self.legal_actions)
            current = time_step(next_observations[slot], rewards[slot], done, self.legal_actions)
            for name, agent in self.agents.items():
                q = agent._rl_agent
                q.add_transition(previous, joint[name], current)
                q._iteration += 1
                agent._iteration += 1
                if q._iteration % q._learn_every == 0:
                    q._last_loss_value = q.learn()
                    if q._last_loss_value is not None:
                        self._record_loss(name, "rl", q._last_loss_value)
                    if agent._reservoir_buffer is not None:
                        agent._last_sl_loss_value = agent._learn()
                        if agent._last_sl_loss_value is not None:
                            self._record_loss(name, "sl", agent._last_sl_loss_value)
                if q._iteration % q._update_target_network_every == 0:
                    q._copy_weights(q._tau)

    def _record_loss(self, name, kind, loss):
        if not np.isfinite(loss):
            raise RuntimeError(f"non-finite NFSP {kind} loss for {name}")
        self.updates[name][kind] += 1

    def metrics(self):
        return {name: {"updates": self.updates[name], "losses": agent.loss,
                       "transitions": agent.step_counter,
                       "reservoir_samples": (len(agent._reservoir_buffer)
                                             if agent._reservoir_buffer is not None else 0)}
                for name, agent in self.agents.items()}

    def save(self, directory):
        # Inference artifacts, not a promise to resume the exact random stream or buffers.
        for name, agent in self.agents.items():
            q = agent._rl_agent
            torch.save({"format": "sokurl-openspiel-nfsp-v1", "player": name,
                        "observation_shape": self.shape, "num_actions": self.num_actions,
                        "agent_config": self.config, "metrics": self.metrics()[name],
                        "average_network": agent._avg_network.state_dict(),
                        "q_network": q._q_network.state_dict(),
                        "target_q_network": q._target_q_network.state_dict(),
                        "average_optimizer": agent._optimizer.state_dict(),
                        "q_optimizer": q._optimizer.state_dict()}, Path(directory) / (name + ".pt"))

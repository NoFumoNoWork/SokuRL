"""Collect simultaneous on-policy trajectories for two independent PPO learners."""
import numpy as np
import torch

from soku_rl.env.encoding import AGENTS


def stack_observations(observations, player, slots):
    values = [observations[slot][AGENTS[player]] for slot in slots]
    if isinstance(values[0], dict):
        return {key: np.stack([value[key] for value in values]) for key in values[0]}
    return np.stack(values)


class JointRollouts:
    def __init__(self, env, models, seed):
        if len(models) != 2 or models[0].n_steps != models[1].n_steps:
            raise ValueError("joint PPO needs two learners with the same rollout length")
        self.env, self.models = env, models
        self.slots = tuple(range(env.num_envs))
        self.rng = np.random.default_rng(seed)
        self.observations, _ = env.reset(self.seeds(self.slots))
        self.starts = np.ones(env.num_envs, dtype=bool)
        self.records = []
        self.states = [model._last_lstm_states if hasattr(model, "_last_lstm_states") else None
                       for model in models]

    def seeds(self, slots):
        return {slot: int(self.rng.integers(0, 0xFFFFFFFF)) for slot in slots}

    @torch.no_grad()
    def collect(self):
        for model in self.models:
            model.policy.set_training_mode(False)
            model.rollout_buffer.reset()
        for _ in range(self.models[0].n_steps):
            batches, outputs, previous_states = [], [], list(self.states)
            for player, model in enumerate(self.models):
                batch = stack_observations(self.observations, player, self.slots)
                tensor, _ = model.policy.obs_to_tensor(batch)
                if self.states[player] is None:
                    action, value, log_prob = model.policy(tensor)
                else:
                    starts = torch.as_tensor(self.starts, device=model.device, dtype=torch.float32)
                    action, value, log_prob, self.states[player] = model.policy(
                        tensor, self.states[player], starts)
                batches.append(batch)
                outputs.append((action.cpu().numpy(), value, log_prob))
            actions = {slot: {agent: int(outputs[player][0][slot])
                             for player, agent in enumerate(AGENTS)} for slot in self.slots}
            observations, rewards, terms, truncs, infos = self.env.step(actions)
            dones = np.array([terms[slot][AGENTS[0]] or truncs[slot][AGENTS[0]] for slot in self.slots])
            for player, model in enumerate(self.models):
                action, values, log_probs = outputs[player]
                extra = {} if previous_states[player] is None else {"lstm_states": previous_states[player]}
                model.rollout_buffer.add(batches[player], action.reshape(-1, 1),
                    np.array([rewards[slot][AGENTS[player]] for slot in self.slots]),
                    self.starts, values, log_probs, **extra)
                model.num_timesteps += len(self.slots)
            finished = [slot for slot in self.slots if dones[slot]]
            for slot in finished:
                self.records.append(infos[slot][AGENTS[0]])
            if finished:
                restarted, _ = self.env.reset(self.seeds(finished))
                observations.update(restarted)
            self.observations, self.starts = observations, dones
        for player, model in enumerate(self.models):
            batch = stack_observations(self.observations, player, self.slots)
            tensor, _ = model.policy.obs_to_tensor(batch)
            if self.states[player] is None:
                values = model.policy.predict_values(tensor)
            else:
                values = model.policy.predict_values(tensor, self.states[player].vf,
                    torch.as_tensor(self.starts, device=model.device, dtype=torch.float32))
            model.rollout_buffer.compute_returns_and_advantage(values, self.starts)

    def update(self, targets):
        self.collect()
        for model, target in zip(self.models, targets, strict=True):
            model._update_current_progress_remaining(model.num_timesteps, target)
            model.train()

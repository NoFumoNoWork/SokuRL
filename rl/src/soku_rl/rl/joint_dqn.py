"""Simultaneous game collection for independent off-policy Q learners."""
import numpy as np

from soku_rl.env.encoding import AGENTS
from soku_rl.rl.joint import stack_observations


class JointDQN:
    def __init__(self, env, models, seed):
        if len(models) != 2 or models[0].train_freq != models[1].train_freq:
            raise ValueError("joint DQN requires matching step collection frequencies")
        if models[0].train_freq.unit.value != "step":
            raise ValueError("joint DQN requires step-based collection")
        self.env, self.models = env, models
        self.slots = tuple(range(env.num_envs))
        self.rng = np.random.default_rng(seed)
        self.observations, _ = env.reset(self.seeds(self.slots))
        self.records = []

    def seeds(self, slots):
        return {slot: int(self.rng.integers(0, 0xFFFFFFFF)) for slot in slots}

    def update(self, targets):
        for model in self.models:
            model.policy.set_training_mode(False)
        for _ in range(self.models[0].train_freq.frequency):
            batches, outputs = [], []
            for player, model in enumerate(self.models):
                batch = stack_observations(self.observations, player, self.slots)
                if model.num_timesteps < model.learning_starts:
                    actions = np.array([model.action_space.sample() for _ in self.slots])
                else:
                    actions, _ = model.predict(batch, deterministic=False)
                batches.append(batch)
                outputs.append(actions)
            commands = {slot: {agent: int(outputs[player][slot])
                              for player, agent in enumerate(AGENTS)} for slot in self.slots}
            observations, rewards, terms, truncs, infos = self.env.step(commands)
            dones = np.array([terms[slot][AGENTS[0]] or truncs[slot][AGENTS[0]] for slot in self.slots])
            # Store the actual terminal next observations before resetting games.
            for player, (model, target) in enumerate(zip(self.models, targets, strict=True)):
                model.replay_buffer.add(batches[player], stack_observations(observations, player, self.slots),
                    outputs[player], np.array([rewards[slot][AGENTS[player]] for slot in self.slots]),
                    dones, [infos[slot][AGENTS[player]] for slot in self.slots])
                model.num_timesteps += len(self.slots)
                model._update_current_progress_remaining(model.num_timesteps, target)
                model._on_step()
            finished = [slot for slot in self.slots if dones[slot]]
            self.records.extend(infos[slot][AGENTS[0]] for slot in finished)
            if finished:
                restarted, _ = self.env.reset(self.seeds(finished))
                observations.update(restarted)
            self.observations = observations
        for model in self.models:
            if model.num_timesteps > model.learning_starts:
                steps = model.gradient_steps
                if steps == -1:
                    steps = model.train_freq.frequency * self.env.num_envs
                if steps:
                    model.train(gradient_steps=steps, batch_size=model.batch_size)

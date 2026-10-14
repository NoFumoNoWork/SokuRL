"""Reservoir sampling and supervised average-policy fitting for PPO-based NFSP."""
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from soku_rl.rl.storage import PackedObservation


class AveragePolicy:
    def __init__(self, model, config, seed):
        if min(config["capacity"], config["batch_size"], config["updates"]) < 1:
            raise ValueError("average-policy budgets must be positive")
        self.model, self.config = model, config
        self.rng = np.random.default_rng(seed)
        self.samples = []
        self.seen = 0
        self.updates = 0

    def add(self, observations, actions):
        for observation, action in zip(observations, actions, strict=True):
            self.seen += 1
            sample = (PackedObservation.pack(observation), int(action))
            if len(self.samples) < self.config["capacity"]:
                self.samples.append(sample)
            else:
                index = int(self.rng.integers(self.seen))
                if index < len(self.samples):
                    self.samples[index] = sample

    def fit(self):
        if len(self.samples) < self.config["batch_size"]:
            raise ValueError("the complete PPO rollout must supply an average-policy minibatch")
        policy = self.model.policy
        policy.set_training_mode(True)
        losses = []
        for _ in range(self.config["updates"]):
            indices = self.rng.choice(len(self.samples), self.config["batch_size"], replace=False)
            samples = [self.samples[index] for index in indices]
            observations, _ = policy.obs_to_tensor(np.stack([sample[0].unpack() for sample in samples]))
            actions = torch.as_tensor([sample[1] for sample in samples], device=self.model.device)
            _, log_probs, _ = policy.evaluate_actions(observations, actions)
            loss = -log_probs.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite NFSP supervised loss")
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.model.max_grad_norm)
            policy.optimizer.step()
            self.updates += 1
            losses.append(float(loss.detach()))
        return {"loss": float(np.mean(losses)), "updates": self.updates,
                "seen": self.seen, "samples": len(self.samples)}

    def state(self):
        return {"samples": self.samples, "seen": self.seen, "updates": self.updates,
                "rng": self.rng.bit_generator.state}

    def restore(self, state):
        self.samples, self.seen, self.updates = state["samples"], state["seen"], state["updates"]
        # Version 1 checkpoints stored raw NumPy arrays. Migrate their storage
        # once, keeping the sample order and reservoir random state unchanged.
        self.samples = [(PackedObservation.pack(value) if isinstance(value, np.ndarray) else value, action)
                        for value, action in self.samples]
        if len(self.samples) > self.config["capacity"] or self.seen < len(self.samples):
            raise ValueError("invalid NFSP reservoir checkpoint")
        self.rng.bit_generator.state = state["rng"]


class BestResponseSamples(BaseCallback):
    def __init__(self, average):
        super().__init__()
        self.average = average

    def _on_step(self):
        self.average.add(self.model._last_obs, self.locals["actions"])
        return True

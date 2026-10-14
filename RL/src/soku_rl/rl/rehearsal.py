"""Bounded actor supervision between upstream PPO updates, with fresh recurrent memory."""
import copy
import math
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from soku_rl.rl.recurrent_cloning import demonstration_episodes, zero_states
from soku_rl.rl.sparse_transfer import restore_batch
from soku_rl.rl.demonstration_supervision import supervision_mask


def validate_rehearsal(config):
    keys = {"datasets", "updates_per_rollout", "sequences", "sequence_length", "learning_rate", "seed"}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError("rehearsal requires explicit datasets, update budget, sequence dimensions, learning_rate and seed")
    if (not isinstance(config["datasets"], list) or not config["datasets"]
            or any(not isinstance(path, str) or not path for path in config["datasets"])):
        raise ValueError("rehearsal datasets must be a nonempty list of paths")
    for key in ("updates_per_rollout", "sequences", "sequence_length"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"rehearsal {key} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("rehearsal seed must be a nonnegative integer")
    rate = config["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("rehearsal learning_rate must be finite and positive")


def attach_rehearsal(model, interface, config, continuing):
    from soku_rl.rl.demonstration_sets import load_demonstration_sets
    validate_rehearsal(config)
    samples, _, identities = load_demonstration_sets(config["datasets"], interface, 0.)
    identity = {"schema": 1, "config": copy.deepcopy(config), "datasets": identities}
    if continuing:
        if not hasattr(model, "rehearsal_state") or model.rehearsal_state["identity"] != identity:
            raise ValueError("continued rehearsal requires identical configuration and verified demonstration data")
    else:
        model.rehearsal_state = {"identity": identity,
            "rng": np.random.default_rng(config["seed"]).bit_generator.state,
            "updates": 0, "frames": 0, "burn_in_frames": 0}
    # Only training games enter the replay store. Held-out games were verified by
    # the strict loader but are never accessible to the sampler.
    model._rehearsal = DemonstrationRehearsal(samples["train"], config)


def window_distribution(model, episode, offset, length):
    """Rebuild every prefix under current weights; truncate gradients only at offset."""
    policy = model.policy
    rows = episode[offset:offset + length]
    observations = restore_batch([row[0] for row in rows], model.device)
    if isinstance(model, RecurrentPPO):
        states = zero_states(policy, 1).pi
        with torch.no_grad():
            for first in range(0, offset, 256):
                prefix = episode[first:min(first + 256, offset)]
                packed = restore_batch([row[0] for row in prefix], model.device)
                _, states = policy.get_distribution(packed, states, torch.zeros(len(prefix), device=model.device))
        # cuDNN's eval RNN forward cannot backpropagate. Use PyTorch's native
        # LSTM for this short differentiable window; keep fast cuDNN inference
        # for the full prefix and preserve eval-mode dropout/BN semantics.
        with torch.backends.cudnn.flags(enabled=False):
            distribution, _ = policy.get_distribution(observations, states,
                torch.zeros(len(rows), device=model.device))
    else:
        distribution = policy.get_distribution(observations)
    actions = torch.as_tensor(np.asarray([row[1] for row in rows]), device=model.device)
    # Consume immediately: SB3 reuses its mutable action-distribution object.
    return -distribution.log_prob(actions), distribution.mode() == actions


class DemonstrationRehearsal:
    def __init__(self, samples, config):
        self.episodes = demonstration_episodes(samples)
        self.positions = [np.flatnonzero(supervision_mask(episode)) for episode in self.episodes]
        if any(len(indices) and not np.array_equal(indices, np.arange(indices[0], len(episode)))
               for indices, episode in zip(self.positions, self.episodes, strict=True)):
            raise ValueError("rehearsal supervision must be a contiguous suffix of each episode")
        self.ends = np.cumsum([len(indices) for indices in self.positions])
        if not self.ends[-1]:
            raise ValueError("rehearsal requires supervised demonstration frames")
        self.config = copy.deepcopy(config)

    def update(self, model):
        started = time.perf_counter()
        config, state = self.config, model.rehearsal_state
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        frames, burn_in, correct, nll_sum = 0, 0, 0, 0.
        policy = model.policy
        # eval mode disables dropout/BN mutations in both prefix and supervised
        # window; gradients remain enabled in windows.
        policy.set_training_mode(False)
        rates = [group["lr"] for group in policy.optimizer.param_groups]
        try:
            for group in policy.optimizer.param_groups:
                group["lr"] = config["learning_rate"]
            for _ in range(config["updates_per_rollout"]):
                policy.optimizer.zero_grad(set_to_none=True)
                windows = []
                for index in rng.integers(int(self.ends[-1]), size=config["sequences"]):
                    episode_index = int(np.searchsorted(self.ends, index, side="right"))
                    position = int(index - (0 if episode_index == 0 else self.ends[episode_index - 1]))
                    offset = int(self.positions[episode_index][position])
                    episode = self.episodes[episode_index]
                    size = min(config["sequence_length"], len(episode) - offset)
                    windows.append((episode, offset, size))
                count = sum(size for _, _, size in windows)
                # Accumulate gradients before stepping so every prefix/window
                # sees identical weights. Prefix graphs are never retained.
                for episode, offset, size in windows:
                    nll, matches = window_distribution(model, episode, offset, size)
                    loss = nll.sum() / count
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite rehearsal loss")
                    loss.backward()
                    nll_sum += float(nll.detach().sum())
                    correct += int(matches.sum())
                    frames += size
                    burn_in += offset if isinstance(model, RecurrentPPO) else 0
                torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm, error_if_nonfinite=True)
                policy.optimizer.step()
                state["updates"] += 1
            state["rng"] = rng.bit_generator.state
            state["frames"] += frames
            state["burn_in_frames"] += burn_in
        finally:
            for group, rate in zip(policy.optimizer.param_groups, rates, strict=True):
                group["lr"] = rate
        metrics = {"nll": nll_sum / frames, "accuracy": correct / frames,
            "frames": frames, "burn_in_frames": burn_in, "updates": state["updates"],
            "total_frames": state["frames"], "ppo_steps": model.num_timesteps,
            "seconds": time.perf_counter() - started}
        state["last_update"] = metrics
        for key, value in metrics.items():
            model.logger.record(f"rehearsal/{key}", value)
        return metrics


class RehearsalUpdates:
    def train(self):
        if not hasattr(self, "_rehearsal"):
            raise RuntimeError("rehearsal training must use the shared PPO factory to verify and attach data")
        super().train()
        self._rehearsal.update(self)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_rehearsal"]


class RehearsalPPO(RehearsalUpdates, PPO):
    pass


class RehearsalRecurrentPPO(RehearsalUpdates, RecurrentPPO):
    pass


def rehearsal_algorithm(algorithm):
    if algorithm is PPO:
        return RehearsalPPO
    if algorithm is RecurrentPPO:
        return RehearsalRecurrentPPO
    raise ValueError("rehearsal requires the shared feedforward or recurrent PPO")

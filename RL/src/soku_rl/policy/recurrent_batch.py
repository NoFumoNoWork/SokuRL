"""Batch frozen recurrent inference while retaining each episode's memory and RNG."""
import numpy as np
import torch

from soku_rl.policy.batch import episode_actions
from soku_rl.policy.recurrent import RecurrentEpisode


@torch.inference_mode()
def recurrent_group_actions(entries):
    policy = entries[0][1].model.policy
    observations = [observation for _, _, observation in entries]
    if isinstance(observations[0], dict):
        batch = {key: np.stack([observation[key] for observation in observations]) for key in observations[0]}
    else:
        batch = np.stack(observations)
    tensor, _ = policy.obs_to_tensor(batch)
    states = tuple(torch.cat([actor.states[index] for _, actor, _ in entries], dim=1) for index in (0, 1))
    starts = torch.cat([actor.start for _, actor, _ in entries])
    distribution, states = policy.get_distribution(tensor, states, starts)
    probabilities = distribution.distribution.probs.cpu().numpy().astype(np.float64)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("recurrent policy produced non-finite probabilities")
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    actions = {}
    for index, (key, actor, _) in enumerate(entries):
        # Own storage prevents an episode retaining another actor's memory.
        actor.states = tuple(value[:, index:index + 1].clone() for value in states)
        actor.start.zero_()
        row = probabilities[index]
        actions[key] = int(actor.rng.choice(len(row), p=row))
    return actions


def recurrent_episode_actions(requests):
    groups, other, seen = {}, {}, set()
    for key, (actor, observation) in requests.items():
        # Subclasses or wrappers can override act(); preserve their own behavior.
        if type(actor) is RecurrentEpisode:
            if id(actor) in seen:
                raise ValueError("a recurrent actor may appear only once in a batch")
            seen.add(id(actor))
            groups.setdefault(id(actor.model), []).append((key, actor, observation))
        else:
            other[key] = actor, observation
    actions = episode_actions(other)
    for entries in groups.values():
        actions.update(recurrent_group_actions(entries))
    return actions

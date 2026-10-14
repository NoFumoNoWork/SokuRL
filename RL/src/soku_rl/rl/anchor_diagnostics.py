"""Measure frozen-policy distance on explicit episode windows, without game interaction."""
import torch
from sb3_contrib import RecurrentPPO

from soku_rl.rl.actor_windows import actor_window_logits, categorical_distance
from soku_rl.rl.recurrent_cloning import zero_states
from soku_rl.rl.sparse_transfer import restore_batch


def select_windows(episodes, offsets, length):
    if (not episodes or any(not episode for episode in episodes)
            or not isinstance(offsets, list) or not offsets
            or any(type(value) is not int or value < 0 for value in offsets)
            or len(set(offsets)) != len(offsets) or type(length) is not int or length < 1):
        raise ValueError('diagnostic windows require nonempty episodes, unique nonnegative offsets and positive length')
    windows = [(index, offset, length) for index, episode in enumerate(episodes)
        for offset in offsets if offset + length <= len(episode)]
    if not windows:
        raise ValueError('no complete diagnostic window fits the episodes')
    return windows


def reference_windows(reference, episodes, windows):
    reference.policy.set_training_mode(False)
    return [actor_window_logits(reference, episodes[index], offset, length, False).detach()
        for index, offset, length in windows]


def score_windows(model, episodes, windows, targets):
    if len(windows) != len(targets) or not windows:
        raise ValueError('each diagnostic window requires a reference distribution')
    model.policy.set_training_mode(False)
    rows = []
    for (index, offset, length), target in zip(windows, targets, strict=True):
        current = actor_window_logits(model, episodes[index], offset, length, False)
        kl, tv = categorical_distance(target, current)
        if not torch.isfinite(kl).all() or not torch.isfinite(tv).all():
            raise RuntimeError('non-finite diagnostic distribution distance')
        rows.append({'episode_index': index, 'offset': offset, 'frames': length,
            'kl_by_frame': kl.cpu().tolist(), 'tv_by_frame': tv.cpu().tolist(),
            'argmax_agreement': int((target.argmax(-1) == current.argmax(-1)).sum())})
    frames = sum(row['frames'] for row in rows)
    return {'frames': frames, 'mean_kl': sum(sum(row['kl_by_frame']) for row in rows) / frames,
        'mean_tv': sum(sum(row['tv_by_frame']) for row in rows) / frames,
        'argmax_agreement': sum(row['argmax_agreement'] for row in rows) / frames, 'windows': rows}


def replay_actor_state(model, episode, offset):
    if not isinstance(model, RecurrentPPO) or not 0 <= offset <= len(episode):
        raise ValueError('recurrent memory diagnostic requires a valid episode prefix')
    model.policy.set_training_mode(False)
    states = zero_states(model.policy, 1).pi
    with torch.no_grad():
        for first in range(0, offset, 256):
            batch = restore_batch(episode[first:min(first + 256, offset)], model.device)
            _, states = model.policy.get_distribution(batch, states, torch.zeros(len(batch), device=model.device))
    return states


def score_memory_transfer(model, reference, episodes, windows):
    """Keep current weights fixed, replacing only the initial window memory by the reference's."""
    model.policy.set_training_mode(False)
    rows = []
    for index, offset, length in windows:
        episode = episodes[index]
        current = actor_window_logits(model, episode, offset, length, False)
        memory = replay_actor_state(reference, episode, offset)
        batch = restore_batch(episode[offset:offset + length], model.device)
        model.policy.set_training_mode(False)
        with torch.no_grad():
            distribution, _ = model.policy.get_distribution(batch, memory, torch.zeros(length, device=model.device))
            transferred = distribution.distribution.logits.clone()
        _, tv = categorical_distance(current, transferred)
        if offset == 0 and not torch.equal(current, transferred):
            raise RuntimeError('zero-prefix memory control differs')
        rows.append({'episode_index': index, 'offset': offset, 'frames': length,
            'tv_by_frame': tv.cpu().tolist()})
    return rows

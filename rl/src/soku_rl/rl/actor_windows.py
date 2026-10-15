"""Evaluate actor distributions with independent, freshly replayed episode memory."""
import torch
from sb3_contrib import RecurrentPPO

from soku_rl.rl.recurrent_cloning import zero_states
from soku_rl.rl.sparse_transfer import restore_batch


def actor_window_logits(model, episode, offset, length, gradients):
    if offset < 0 or length < 1 or offset + length > len(episode):
        raise ValueError("actor window must fit within one observed episode")
    policy = model.policy
    observations = restore_batch(episode[offset:offset + length], model.device)
    with torch.set_grad_enabled(gradients):
        if isinstance(model, RecurrentPPO):
            states = zero_states(policy, 1).pi
            with torch.no_grad():
                for first in range(0, offset, 256):
                    prefix = restore_batch(episode[first:min(first + 256, offset)], model.device)
                    _, states = policy.get_distribution(prefix, states, torch.zeros(len(prefix), device=model.device))
            # Native LSTM supports gradients in eval mode; keep dropout/BN off.
            with torch.backends.cudnn.flags(enabled=False if gradients else torch.backends.cudnn.enabled):
                distribution, _ = policy.get_distribution(observations, states,
                    torch.zeros(length, device=model.device))
        else:
            distribution = policy.get_distribution(observations)
        # SB3 reuses its mutable distribution container. Return independent data.
        return distribution.distribution.logits.clone()


def categorical_distance(reference, current):
    if reference.shape != current.shape or reference.ndim != 2:
        raise ValueError("anchor distributions must have matching frame/action dimensions")
    # Re-normalize in double precision so a near-zero KL is not dominated by
    # float32 normalization error. Network computation retains its original dtype.
    p = torch.distributions.Categorical(logits=reference.double())
    q = torch.distributions.Categorical(logits=current.double())
    return torch.distributions.kl_divergence(p, q), .5 * (p.probs - q.probs).abs().sum(-1)

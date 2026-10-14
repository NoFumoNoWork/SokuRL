"""Initial categorical logits for full logical actions; never mask actions."""
import numpy as np


def logical_action_prior(interface, settings):
    if set(settings) != {"button_probability"}:
        raise ValueError("initial_action_prior requires button_probability")
    probability = settings["button_probability"]
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not np.isfinite(probability) or not 0 < probability < 1):
        raise ValueError("button_probability must be finite and strictly between zero and one")
    if interface.commands != tuple(range(576)):
        raise ValueError("logical action initialization requires all 576 actions")
    buttons = np.arange(576, dtype=np.int64) % 64
    counts = ((buttons[:, None] >> np.arange(6)) & 1).sum(1)
    logits = counts * np.log(probability) + (6 - counts) * np.log1p(-probability)
    # Every direction is equally likely. Centering leaves the softmax unchanged.
    return (logits - logits.mean()).astype(np.float32)


def initialize_action_bias(model, logits):
    import torch
    bias = model.policy.action_net.bias
    if tuple(bias.shape) != tuple(logits.shape):
        raise ValueError("action head does not match logical categorical prior")
    with torch.no_grad():
        bias.add_(torch.as_tensor(logits, device=bias.device, dtype=bias.dtype))

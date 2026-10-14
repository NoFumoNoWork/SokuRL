"""Actor supervision with explicit emphasis on changes from the previous command."""
import math

import torch


def validate_change_weight(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 1:
        raise ValueError("action_change_weight must be finite and at least one")


def weighted_action_loss(nll, changed, valid, change_weight):
    validate_change_weight(change_weight)
    if (nll.ndim != 1 or changed.shape != nll.shape or valid.shape != nll.shape
            or changed.dtype != torch.bool or valid.dtype != torch.bool or not valid.any()):
        raise ValueError("actor loss requires aligned frame vectors and a nonempty validity mask")
    if change_weight == 1:
        return nll[valid].mean()
    weights = torch.where(changed[valid], change_weight, 1.).to(nll.dtype)
    return (nll[valid] * weights).sum() / weights.sum()


def weighted_validation_nll(metrics, frames, change_weight):
    validate_change_weight(change_weight)
    if type(frames) is not int or frames < 1 or not 0 <= metrics['changed_samples'] <= frames:
        raise ValueError("validation requires positive frame count and consistent change count")
    if change_weight == 1 or metrics['changed_samples'] == 0:
        return metrics['nll']
    extra = (change_weight - 1) * metrics['changed_samples']
    return (frames * metrics['nll'] + extra * metrics['changed_nll']) / (frames + extra)

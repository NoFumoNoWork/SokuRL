"""Teacher-label diagnostics for logical buttons, not confirmed game actions."""
import torch


def command_group_totals(distribution, labels, valid):
    probabilities = distribution.distribution.probs.detach()
    if probabilities.ndim != 2 or probabilities.shape[1] != 576 or not bool(valid.any()):
        raise ValueError("command diagnostics require valid frames and all 576 categorical commands")
    nll = -distribution.log_prob(labels)[valid].detach()
    probabilities, labels = probabilities[valid], labels[valid]
    predicted = probabilities.argmax(dim=1)
    commands = torch.arange(576, device=labels.device)
    totals = {"frames": int(labels.numel())}
    for name, bits in (("attack", 7), ("change_card", 16), ("spellcard", 32)):
        selected = (commands & bits) != 0
        positive = (labels & bits) != 0
        mode_positive = (predicted & bits) != 0
        mass = probabilities[:, selected].sum(dim=1)
        totals.update({f"{name}_labels": int(positive.sum()),
            f"{name}_mode": int(mode_positive.sum()), f"{name}_expected": float(mass.sum()),
            f"{name}_correct": int(((predicted == labels) & positive).sum()),
            f"{name}_recalled": int((mode_positive & positive).sum()),
            f"{name}_label_mass": float(mass[positive].sum()),
            f"{name}_nll": float(nll[positive].sum())})
    return totals


def summarize_command_groups(totals):
    if not totals or totals["frames"] < 1:
        raise ValueError("command diagnostics require nonempty totals")
    metrics = {}
    for name in ("attack", "change_card", "spellcard"):
        labels = totals[f"{name}_labels"]
        metrics.update({f"{name}_label_frames": labels,
            f"{name}_label_rate": labels / totals["frames"],
            f"{name}_mode_rate": totals[f"{name}_mode"] / totals["frames"],
            f"{name}_expected_rate": totals[f"{name}_expected"] / totals["frames"]})
        if labels:
            metrics.update({f"{name}_exact_accuracy": totals[f"{name}_correct"] / labels,
                f"{name}_mode_recall": totals[f"{name}_recalled"] / labels,
                f"{name}_expected_on_labels": totals[f"{name}_label_mass"] / labels,
                f"{name}_label_nll": totals[f"{name}_nll"] / labels})
    return metrics

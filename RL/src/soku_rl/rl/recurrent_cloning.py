"""Truncated sequence supervision for the existing shared RecurrentPPO policy."""
import numpy as np
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from soku_rl.rl.sparse_transfer import restore_batch
from soku_rl.rl.command_diagnostics import command_group_totals, summarize_command_groups
from soku_rl.rl.demonstration_loss import validate_change_weight, weighted_action_loss
from soku_rl.rl.demonstration_supervision import supervision_mask


def demonstration_episodes(samples):
    episodes = []
    for row in samples:
        if row[3] == -1:
            episodes.append([])
        elif row[3] not in (0, 1) or not episodes:
            raise ValueError("recurrent demonstrations require explicit whole-episode boundaries")
        episodes[-1].append(row)
    if not episodes:
        raise ValueError("recurrent demonstrations must be nonempty")
    return episodes


def episode_chunks(episodes, order, sequence_length, max_sequences):
    """Pad only sequence tails; preserve order and identify surviving state columns."""
    if not episodes or sorted(order) != list(range(len(episodes))) or any(not episode for episode in episodes):
        raise ValueError("sequence order must include each nonempty episode exactly once")
    if (type(sequence_length) is not int or sequence_length < 1
            or type(max_sequences) is not int or max_sequences < 1):
        raise ValueError("sequence dimensions must be positive integers")
    for first in range(0, len(order), max_sequences):
        group = [episodes[int(index)] for index in order[first:first + max_sequences]]
        previous = list(range(len(group)))
        for offset in range(0, max(map(len, group)), sequence_length):
            active = [index for index, episode in enumerate(group) if offset < len(episode)]
            chunks = [group[index][offset:offset + sequence_length] for index in active]
            width = max(map(len, chunks))
            rows, valid = [], []
            for chunk in chunks:
                rows.extend([*chunk, *([chunk[-1]] * (width - len(chunk)))])
                valid.extend([True] * len(chunk) + [False] * (width - len(chunk)))
            yield {"new_group": offset == 0, "keep": [previous.index(index) for index in active],
                "samples": rows, "valid": valid}
            previous = active


def zero_states(policy, sequences):
    shape = (policy.lstm_hidden_state_shape[0], sequences, policy.lstm_hidden_state_shape[2])
    return RNNStates(*(tuple(torch.zeros(shape, device=policy.device) for _ in range(2)) for _ in range(2)))


def sequence_epoch(model, episodes, order, batch_size, sequence_length, value_coef, training, action_change_weight):
    validate_change_weight(action_change_weight)
    if (type(batch_size) is not int or type(sequence_length) is not int or sequence_length < 1
            or batch_size < sequence_length or batch_size % sequence_length):
        raise ValueError("recurrent batch_size must be a positive multiple of sequence_length")
    model.policy.set_training_mode(training)
    totals = dict(nll=0., accuracy=0., value_mse=0., entropy=0.)
    count, changed_count, changed_correct, changed_nll = 0, 0, 0., 0.
    updates, total_loss = 0, 0.
    command_totals = {}
    with torch.set_grad_enabled(training):
        for chunk in episode_chunks(episodes, order, sequence_length, batch_size // sequence_length):
            if chunk["new_group"]:
                states = zero_states(model.policy, len(chunk["keep"]))
            else:
                states = RNNStates(*(tuple(value[:, chunk["keep"], :] for value in pair) for pair in states))
            rows = chunk["samples"]
            observations = restore_batch([row[0] for row in rows], model.device)
            actions = torch.as_tensor(np.asarray([row[1] for row in rows]), device=model.device)
            returns = torch.as_tensor(np.asarray([row[2] for row in rows], dtype=np.float32), device=model.device)
            valid = torch.as_tensor(np.asarray(chunk["valid"]) & supervision_mask(rows), device=model.device)
            # Every new episode column is explicitly zeroed above; no resets
            # occur inside a chunk. SB3 can therefore use its batched LSTM path.
            starts = torch.zeros(len(rows), device=model.device)
            with torch.set_grad_enabled(training and bool(valid.any())):
                predicted, values, _, next_states = model.policy(observations, states, starts, deterministic=True)
            if not valid.any():
                states = RNNStates(*(tuple(value.detach() for value in pair) for pair in next_states))
                continue
            # forward() populated the same SB3 action distribution used online.
            distribution = model.policy.action_dist
            nll = -distribution.log_prob(actions)
            squared_error = (values.flatten() - returns) ** 2
            entropy = distribution.entropy()
            changed = torch.as_tensor([row[3] == 1 for row in rows], device=model.device)
            if action_change_weight == 1:
                loss = (nll[valid] + value_coef * squared_error[valid]).mean()
            else:
                loss = weighted_action_loss(nll, changed, valid, action_change_weight) + value_coef * squared_error[valid].mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite recurrent cloning loss")
            if training:
                model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                model.policy.optimizer.step()
                updates += 1
            # Carry memory, but truncate gradients at chunk boundaries. Training
            # states precede the latest parameter update, as in online TBPTT.
            states = RNNStates(*(tuple(value.detach() for value in pair) for pair in next_states))
            size = int(valid.sum())
            count += size
            total_loss += float(loss.detach()) * size
            correct = predicted == actions
            totals["nll"] += float(nll[valid].detach().sum())
            totals["accuracy"] += float(correct[valid].sum())
            totals["value_mse"] += float(squared_error[valid].detach().sum())
            totals["entropy"] += float(entropy[valid].detach().sum())
            changed = changed & valid
            changed_count += int(changed.sum())
            changed_correct += float(correct[changed].sum())
            changed_nll += float(nll[changed].detach().sum())
            if not training and model.action_space.n == 576:
                diagnostics = command_group_totals(distribution, actions, valid)
                for key, value in diagnostics.items():
                    command_totals[key] = command_totals.get(key, 0) + value
    if not count:
        raise ValueError("sequence scoring requires supervised demonstration frames")
    result = {key: value / count for key, value in totals.items()}
    result["changed_samples"] = changed_count
    if changed_count:
        result.update(changed_accuracy=changed_correct / changed_count, changed_nll=changed_nll / changed_count)
    if command_totals:
        result.update(summarize_command_groups(command_totals))
    if not all(np.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite recurrent cloning metrics")
    return result, total_loss / count, updates

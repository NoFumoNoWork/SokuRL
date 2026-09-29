"""Fit OpenSpiel response networks with bounded Double DQN targets."""
import torch


def double_q_targets(rewards, final, online, target, legal, discount, bound):
    if not all(torch.isfinite(value).all() for value in (rewards, online, target)):
        raise ValueError("response target inputs must be finite")
    if not legal[~final].any(-1).all():
        raise ValueError("an ongoing transition must have a legal next action")
    chosen = online.masked_fill(~legal, -torch.inf).argmax(-1, keepdim=True)
    following = target.gather(-1, chosen).squeeze(-1)
    raw = torch.where(final, rewards, rewards + discount * following)
    return raw.clamp(-bound, bound), raw


def learn_response(agent, return_bound):
    """Use the pinned OpenSpiel replay, optimizer, loss and target network."""
    transitions = agent._replay_buffer.sample(agent._batch_size)

    def tensor(value, dtype):
        return torch.as_tensor(value, device=agent._device, dtype=dtype)

    states = tensor(transitions.info_state, torch.float32)
    actions = tensor(transitions.action, torch.long)
    rewards = tensor(transitions.reward, torch.float32)
    following = tensor(transitions.next_info_state, torch.float32)
    final = tensor(transitions.is_final_step, torch.bool)
    legal = tensor(transitions.legal_actions_mask, torch.bool)
    with torch.no_grad():
        targets, raw = double_q_targets(rewards, final, agent._q_network(following),
            agent._target_q_network(following), legal, agent._discount_factor, return_bound)
    values = agent._q_network(states).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    loss = agent.loss_class(values, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite response loss")
    agent._optimizer.zero_grad()
    loss.backward()
    if agent._gradient_norm_clipping is not None:
        torch.nn.utils.clip_grad_norm_(agent._q_network.parameters(), agent._gradient_norm_clipping)
    agent._optimizer.step()
    return loss.item(), {"prediction_min": values.detach().min().item(),
        "prediction_max": values.detach().max().item(), "target_min": targets.min().item(),
        "target_max": targets.max().item(), "clipped_fraction": (targets != raw).float().mean().item()}

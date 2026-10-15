"""Double DQN using SB3's collector, target synchronization and optimizer lifecycle."""
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import DQN


def double_q_target(online, target, samples, gamma):
    """Select with the online network, evaluate with the frozen target network."""
    with torch.no_grad():
        actions = online(samples.next_observations).argmax(dim=1, keepdim=True)
        values = target(samples.next_observations).gather(1, actions)
        discounts = gamma if samples.discounts is None else samples.discounts
        return samples.rewards + (1 - samples.dones) * discounts * values


class DoubleDQN(DQN):
    def _on_step(self):
        super()._on_step()
        fraction = min(self.num_timesteps / self.exploration_decay_steps, 1.)
        self.exploration_rate = (self.exploration_initial_eps
            + fraction * (self.exploration_final_eps - self.exploration_initial_eps))
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps, batch_size):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses, errors, values = [], [], []
        for _ in range(gradient_steps):
            samples = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            targets = double_q_target(self.q_net, self.q_net_target, samples, self.gamma)
            estimates = self.q_net(samples.observations).gather(1, samples.actions.long())
            loss = F.smooth_l1_loss(estimates, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Double DQN loss")
            self.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm,
                                           error_if_nonfinite=True)
            self.policy.optimizer.step()
            # Count completed optimizer steps even if a later update in this
            # block fails and BR saves an interrupted recovery checkpoint.
            self._n_updates += 1
            losses.append(loss.detach())
            errors.append((estimates.detach() - targets).abs().mean())
            values.append(estimates.detach().mean())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if losses:
            stats = torch.stack([torch.stack(items).mean() for items in (losses, errors, values)]).cpu().tolist()
            for key, value in zip(("loss", "td_error", "q_mean"), stats, strict=True):
                self.logger.record("train/" + key, value)

    def predict(self, observation, *args, **kwargs):
        # Upstream uses one epsilon coin for the whole vector batch. Independent
        # coins avoid synchronized exploration across independent game instances.
        deterministic = kwargs.get("deterministic", args[2] if len(args) > 2 else False)
        action, state = self.policy.predict(observation, *args, **kwargs)
        if not deterministic:
            action = np.asarray(action).copy()
            mask = np.random.random(action.shape) < self.exploration_rate
            random_actions = np.random.randint(self.action_space.n, size=action.shape)
            action = np.where(mask, random_actions, action)
        return action, state

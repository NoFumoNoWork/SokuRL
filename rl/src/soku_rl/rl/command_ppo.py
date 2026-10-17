"""Duration-aware sampling adapter; the PPO clipped objective stays SB3's."""
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from gymnasium import spaces

from soku_rl.env.encoding import AGENTS
from soku_rl.env.v3c import CommandExecution, V3Ledger, vocabulary
from soku_rl.rl.matchup_env import MatchupMixtureVecEnv


class DurationRolloutBuffer(RolloutBuffer):
    def reset(self):
        super().reset()
        self.durations = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.censored = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.detach().cpu().numpy().flatten()
        gae = np.zeros(self.n_envs)
        for step in reversed(range(self.buffer_size)):
            live = 1 - (dones.astype(np.float32) if step == self.buffer_size - 1
                        else self.episode_starts[step + 1])
            nxt = last_values if step == self.buffer_size - 1 else self.values[step + 1]
            discount = self.gamma ** self.durations[step]
            delta = self.rewards[step] + discount * nxt * live - self.values[step]
            gae = delta + (self.gamma * self.gae_lambda) ** self.durations[step] * live * gae
            # A budget-cut command has no observed next decision boundary.
            # Exclude it, and bootstrap preceding decisions at its START value.
            gae = np.where(self.censored[step], 0., gae)
            self.advantages[step] = gae
        self.returns = self.advantages + self.values

    def get(self, batch_size=None):
        # Zero-duration rows are frozen, budget-exhausted slots, NOT decisions.
        assert self.full
        valid = self.swap_and_flatten(self.durations[..., None]).ravel() > 0
        valid &= ~self.swap_and_flatten(self.censored[..., None]).ravel()
        if not self.generator_ready:
            for name in ('observations', 'actions', 'values', 'log_probs', 'advantages', 'returns'):
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        indices = np.random.permutation(np.flatnonzero(valid))
        batch_size = len(indices) if batch_size is None else batch_size
        for start in range(0, len(indices), batch_size):
            yield self._get_samples(indices[start:start + batch_size])


class CommandMatchupEnv(MatchupMixtureVecEnv):
    """Two active seats, different macro lengths, opponent acts every native frame."""
    def __init__(self, *args, native_budget, gamma, state_reader, v3=False, **kwargs):
        super().__init__(*args, **kwargs)
        if native_budget % self.num_envs or native_budget < self.num_envs:
            raise ValueError('native budget must split equally across slots')
        self.frame_target = native_budget // self.num_envs
        self.native_frames = np.zeros(self.num_envs, dtype=np.int64)
        self.decisions = np.zeros(self.num_envs, dtype=np.int64)
        self.gamma, self.state_reader, self.v3 = gamma, state_reader, v3
        self.programs = vocabulary()
        if v3:
            self.action_space = spaces.Discrete(len(self.programs))
        self.ledgers = {}
        self.ledger_factory = V3Ledger
        self.episode_stats = []
        self.frame_records = []
        self.last_durations = np.zeros(self.num_envs, dtype=np.int32)
        self.auto_reset = True
        self.completed_slots = set()
        self.interaction = [dict(approach_frames=0, close_frames=0, total_frames=0) for _ in range(self.num_envs)]

    def _reset_slots(self, seeds):
        super()._reset_slots(seeds)
        for slot in seeds:
            self.ledgers[slot] = self.ledger_factory()
            self.completed_slots.discard(slot)

    @property
    def finished(self):
        return bool((self.native_frames >= self.frame_target).all())

    def step_wait(self):
        if self.pending is None:
            raise RuntimeError('missing learner decision')
        pending, self.pending = self.pending, None
        active = {s for s in range(self.num_envs) if self.native_frames[s] < self.frame_target
                  and s not in self.completed_slots}
        durations = np.zeros(self.num_envs, dtype=np.int32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        output = [dict(valid_decision=False, duration=0) for _ in range(self.num_envs)]
        commands = {}
        if self.v3:
            for slot in active:
                commands[slot] = CommandExecution(self.programs[int(pending[slot])][1],
                                                    self.state_reader(slot, self.players[slot]))
        while active:
            before = {s: tuple(self.state_reader(s, p) for p in (0, 1)) for s in active}
            joint = {}
            for s in active:
                seat = self.players[s]
                command = (self.env.interface.action(commands[s].action(
                    self.env.episodes[s].frame, before[s][seat])) if self.v3 else int(pending[s]))
                joint[s] = {AGENTS[seat]: command, AGENTS[1 - seat]: self.actors[s].act(
                    self.observations[s][AGENTS[1 - seat]])}
            obs, values, terms, truncs, infos = self.env.step(joint)
            finished = set()
            for s in active:
                seat, agent = self.players[s], AGENTS[self.players[s]]
                after = tuple(self.state_reader(s, p) for p in (0, 1))
                if before[s][seat]['frame'] + 1 != after[seat]['frame']:
                    raise RuntimeError('macro lost or repeated a native game frame')
                measures = self.interaction[s]
                measures['total_frames'] += 1
                old_distance = abs(before[s][0]['x'] - before[s][1]['x'])
                distance = abs(after[0]['x'] - after[1]['x'])
                measures['close_frames'] += int(distance <= 160.)
                measures['approach_frames'] += int(distance < old_distance - .01 and
                    (after[seat]['x'] - before[s][seat]['x']) * before[s][seat]['facing'] > .01)
                info = dict(infos[s][agent])
                events = info['native_combat_events']
                parts = dict(info['combat_reward_parts'])
                value = values[s][agent]
                extra = self.ledgers[s].step(events, before[s], after)[seat]
                if self.v3:
                    parts.update(extra)
                    value += sum(extra.values()) / self.env.interface.episode.combat_reward['scale']
                    commands[s].observe(events, seat, after[seat])
                rewards[s] += self.gamma ** durations[s] * value
                self.returns[s] += value
                self.base_returns[s] += info['base_reward']
                self.lengths[s] += 1
                durations[s] += 1
                self.native_frames[s] += 1
                self.observations[s] = obs[s]
                done = terms[s][agent] or truncs[s][agent]
                self.frame_records.append(dict(slot=s, frame=info['frame'], parts=parts,
                    issued_learning_actions=joint[s],
                    events=events, combat=info['combat_reward_summary'],
                    before=before[s], after=after, outcome=info['outcome'],
                    learner_seat=seat, episode=self.episode_context[s], done=bool(done)))
                if done or not self.v3 or commands[s].done or self.native_frames[s] >= self.frame_target:
                    finished.add(s)
                    dones[s] = done
                    info.update(duration=int(durations[s]), valid_decision=True,
                        learner_seat=seat, **{'TimeLimit.truncated': False},
                        command=self.programs[int(pending[s])][0] if self.v3 else int(pending[s]))
                    if self.v3:
                        info['command_feedback'] = dict(accepted=commands[s].accepted,
                            interrupted=commands[s].interrupted, cancel_open=commands[s].cancel_open,
                            budget_cut=not done and not commands[s].done)
                    if done:
                        if not self.auto_reset:
                            self.completed_slots.add(s)
                        info['terminal_observation'] = obs[s][agent].copy()
                        info['episode'] = dict(r=float(self.returns[s]), l=int(self.lengths[s]))
                        self.episode_stats.append(dict(slot=s, **info['episode'],
                            outcome=info['outcome'], context=self.episode_context[s],
                            combat=info['combat_reward_summary'],
                            v3=self.ledgers[s].totals[seat]))
                    output[s] = info
            active -= finished
        self.decisions += durations > 0
        self.last_durations = durations
        reset = {s: int(self.rng.integers(0, 0xFFFFFFFF)) for s in range(self.num_envs)
                 if self.auto_reset and dones[s] and self.native_frames[s] < self.frame_target}
        if reset:
            self._reset_slots(reset)
        return self._stack(), rewards, dones, output


class DurationPPO(PPO):
    """SB3 PPO update with SMDP returns and bounded native-frame collection."""
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if env.finished:
            return False
        if self.use_sde or not isinstance(self.action_space, spaces.Discrete):
            raise NotImplementedError('this bounded adapter supports categorical PPO only')
        self.policy.set_training_mode(False)
        rollout_buffer.buffer_size = n_rollout_steps
        rollout_buffer.reset()
        callback.on_rollout_start()
        n_steps = 0
        while n_steps < n_rollout_steps and not env.finished:
            with torch.no_grad():
                actions, values, log_probs = self.policy(obs_as_tensor(self._last_obs, self.device))
            new_obs, rewards, dones, infos = env.step(actions.cpu().numpy())
            valid = env.last_durations > 0
            self.num_timesteps += int(valid.sum())
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            rollout_buffer.durations[n_steps] = env.last_durations
            rollout_buffer.censored[n_steps] = [info.get('command_feedback',{}).get('budget_cut',False)
                                               for info in infos]
            rollout_buffer.add(self._last_obs, actions.cpu().numpy().reshape(-1, 1), rewards,
                               self._last_episode_starts, values, log_probs)
            self._last_obs, self._last_episode_starts = new_obs, dones
            n_steps += 1
        # Final partial rollouts still update; inactive samples are never trained.
        rollout_buffer.buffer_size = n_steps
        for name in ('observations', 'actions', 'rewards', 'episode_starts', 'values',
                     'log_probs', 'advantages', 'returns', 'durations', 'censored'):
            setattr(rollout_buffer, name, getattr(rollout_buffer, name)[:n_steps])
        rollout_buffer.full = True
        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(values, dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self):
        if ((self.rollout_buffer.durations > 0) & ~self.rollout_buffer.censored).any():
            return super().train()

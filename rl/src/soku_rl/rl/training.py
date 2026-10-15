"""Train one SB3 PPO policy per seat against a fixed rule population."""
import json
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from soku_rl.policy.loader import load_policy
from soku_rl.env.combat_metrics import summarize_combat
from soku_rl.rl.episode_metrics import grouped_episode_metrics, summarize_episodes
from soku_rl.rl.ppo import initialize_ppo, parameter_hash
from soku_rl.rl.learner import save_checkpoint


class ResponseCheckpointCallback(CheckpointCallback):
    def __init__(self, directory, save_freq, curriculum):
        super().__init__(save_freq=save_freq, save_path=str(directory / "checkpoints"), name_prefix="ppo")
        self.curriculum = curriculum

    def _on_step(self):
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            self.curriculum.save(self._checkpoint_path(extension="zip"), self.num_timesteps)
        return result


class EpisodeRecords(BaseCallback):
    def __init__(self, directory, checkpoint_every, curriculum, reward_scale=None):
        super().__init__()
        self.directory = directory
        self.records = []
        self.checkpoint_every = checkpoint_every
        self.timings = []
        self.pending_update = False
        self.last_saved_steps = 0
        self.rollout_record_start = 0
        self.curriculum = curriculum
        self.reward_scale=reward_scale
        self.reward_batch=None
        self.reward_batch_index=0
        from soku_rl.rl.combo_telemetry import ComboTelemetry
        self.combo_telemetry=ComboTelemetry()

    def _on_rollout_start(self):
        if self.pending_update:
            self._finish_update()
        self.rollout_started = time.perf_counter()
        self.rollout_record_start = len(self.records)
        if self.reward_scale is not None:
            from soku_rl.rl.reward_batches import RewardBatch
            self.reward_batch=RewardBatch(self.reward_scale,self.num_timesteps)
            self.combo_telemetry.reset_batch()

    def _on_step(self):
        for index,(done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"], strict=True)):
            if self.reward_batch is not None:
                end=None
                if done:
                    if info['source_truncated']:end='timeout'
                    elif info['outcome']=='double_ko':end='draw'
                    elif info['outcome'] in ('p1_win','p2_win'):
                        player=info['training_context']['player']
                        end='win' if info['outcome']==f'p{player+1}_win' else 'loss'
                    else:raise ValueError('unknown terminal reward outcome')
                self.reward_batch.add(info['combat_reward_parts'],self.locals['rewards'][index],end)
                self.combo_telemetry.observe(index,info,bool(done),self.num_timesteps)
            if done:
                self.records.append({key: info[key] for key in (
                    "episode", "frame", "outcome", "decision_frames", "latency_frames", "training_context")})
                self.records[-1]["end_steps"] = self.num_timesteps
                if "combat_metrics" in info:
                    self.records[-1]["combat_metrics"] = info["combat_metrics"]
                if 'combat_reward_summary' in info:
                    self.records[-1]['combat_reward_summary']=info['combat_reward_summary']
                if "input_metrics" in info:
                    self.records[-1]["input_metrics"] = info["input_metrics"]
                if "curriculum_event" in info:
                    self.records[-1]["curriculum_event"] = info["curriculum_event"]
        return True

    def _on_rollout_end(self):
        self.rollout_finished = time.perf_counter()
        self.flush_reward_batch(complete=True)
        self.timings.append({"steps": self.num_timesteps,
            "rollout_seconds": self.rollout_finished - self.rollout_started})
        self.pending_update = True
        metrics = grouped_episode_metrics(self.records)
        rollout_metrics = summarize_episodes(self.records[self.rollout_record_start:])
        # Off-policy learners may collect several rollouts between logger dumps.
        # Missing means in this rollout must not inherit a previous rollout's data.
        for key in tuple(self.logger.name_to_value):
            if key.startswith(("combat/", "inputs/", "v2/")):
                self.logger.record(key, None)
        self.logger.record("combat/episodes", rollout_metrics["episodes"])
        self.logger.record("combat/measured_episodes", rollout_metrics["combat"]["measured_episodes"])
        self.logger.record("combat/action_measured_episodes", rollout_metrics["combat"]["action_measured_episodes"])
        if rollout_metrics["episodes"]:
            self.logger.record("combat/win_rate", rollout_metrics["win_rate"])
        native=[r['combat_reward_summary'] for r in self.records[self.rollout_record_start:]
                if 'combat_reward_summary' in r]
        if native:
            for key in native[0]:self.logger.record(f'v2/{key}',sum(r[key] for r in native)/len(native))
        for section in ("means", "action_means"):
            for key, value in rollout_metrics["combat"].get(section, {}).items():
                self.logger.record(f"combat/{key}", value)
        for key, value in self.curriculum.scalar_metrics().items():
            self.logger.record(key, value)
        self.logger.record("inputs/measured_episodes", rollout_metrics["inputs"]["measured_episodes"])
        for key, value in rollout_metrics["inputs"].get("pooled", {}).items():
            self.logger.record(f"inputs/{key}", value)
        (self.directory / "progress.json").write_text(json.dumps({
            "steps": self.num_timesteps, "episodes": self.records,
            "episode_summary": metrics, "rollout_episode_summary": rollout_metrics,
            "curriculum": self.curriculum.snapshot(),
            "input_summary": metrics["overall"]["inputs"],
            "combat_summary": summarize_combat([record["combat_metrics"] for record in self.records
                if "combat_metrics" in record])}, indent=2), encoding="utf-8")
        self._write_timings("updating")

    def _finish_update(self):
        from soku_rl.rl.dqn import DoubleDQN
        kind = "dqn" if isinstance(self.model, DoubleDQN) else "ppo"
        self.timings[-1].update(update_seconds=time.perf_counter() - self.rollout_finished)
        self.timings[-1][kind + "_n_updates"] = self.model._n_updates
        self.timings[-1]["learner_n_updates"] = self.model._n_updates
        if hasattr(self.model, "rehearsal_state"):
            self.timings[-1]["rehearsal"] = dict(self.model.rehearsal_state["last_update"])
        if hasattr(self.model, "anchor_state"):
            self.timings[-1]["online_anchor"] = dict(self.model.anchor_state["last_update"])
        if hasattr(self.model, "teacher_state"):
            self.timings[-1]["online_teacher"] = dict(self.model.teacher_state["last_update"])
        steps = self.model.num_timesteps
        if self.model._n_updates and (not self.last_saved_steps
                or steps // self.checkpoint_every > self.last_saved_steps // self.checkpoint_every):
            checkpoints = self.directory / "checkpoints"
            checkpoints.mkdir(exist_ok=True)
            path = checkpoints / f"updated_{steps}_steps.zip"
            save_checkpoint(self.model, path)
            self.curriculum.save(path, steps)
            if (self.directory/'reward_batches.jsonl').exists():
                from soku_rl.rl.reward_batches import plot_batches
                plot_batches(self.directory)
            self.last_saved_steps = steps
            self.timings[-1]["updated_checkpoint"] = str(path)
        self.pending_update = False
        self._write_timings("sampling")

    def _write_timings(self, phase):
        (self.directory / "timing.json").write_text(json.dumps({
            "phase": phase, "rollouts": self.timings}, indent=2), encoding="utf-8")

    def _on_training_end(self):
        if self.pending_update:
            self._finish_update()
        self.finish_reward_logging()
        self._write_timings("finished")

    def flush_reward_batch(self,complete=False):
        if self.reward_batch is None or not self.reward_batch.transitions:return
        from soku_rl.rl.reward_batches import append_batch
        self.reward_batch_index+=1
        row=self.reward_batch.row(self.reward_batch_index,self.num_timesteps,complete)
        row['combo_stats']=self.combo_telemetry.snapshot()
        append_batch(self.directory,row)
        for group in ('hp','normalized','normalized_per_transition','outcome_counts'):
            for name,value in row[group].items():
                self.logger.record(f'reward_batch/{group}/{name}',value)
        self.logger.record('reward_batch/transitions',row['transitions'])
        self.logger.record('reward_batch/residual',row['reward_residual'])
        for key,value in row['combo_stats'].items():
            if isinstance(value,(int,float)) or value is None:
                self.logger.record(f'combo_batch/{key}',value)
        self.reward_batch=None

    def finish_reward_logging(self):
        self.flush_reward_batch()
        if (self.directory/'reward_batches.jsonl').exists():
            from soku_rl.rl.reward_batches import plot_batches
            plot_batches(self.directory)


def train_ppo(env, config, device, seed, directory):
    from soku_rl.rl import validate_payoff
    validate_payoff(env.interface,config)
    if config["players"] != [0, 1]:
        raise ValueError("train both seat policies for seat-swapped evaluation")
    if config["timesteps_per_player"] < 1 or config["checkpoint_every"] < env.num_envs:
        raise ValueError("invalid PPO training or checkpoint interval")
    if not config["opponents"] or len(set(config["opponents"])) != len(config["opponents"]):
        raise ValueError("fixed opponent roster must be nonempty and distinct")
    opponents = [load_policy(name, {"kind": "rule", "name": name, "rules": config["rules"]},
                             env.interface, device) for name in config["opponents"]]
    weights = np.full(len(opponents), 1 / len(opponents))
    results = {}
    for player in config["players"]:
        destination = directory / f"player_{player}"
        destination.mkdir()
        from soku_rl.marl.br import train_response
        response = config | {"player": player, "timesteps": config["timesteps_per_player"],
            "initial_policy": config["initial_policies"][f"player_{player}"], "matchups": {"mode": "fixed"}}
        results[f"player_{player}"] = train_response(
            env, response, opponents, weights, device, seed + player, destination)
    return results

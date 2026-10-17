"""Bounded V2/V3-C native-frame comparison, no training by default."""
import argparse
from copy import deepcopy
from contextlib import ExitStack, closing
from dataclasses import asdict
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from sokurl_platform.bridge_shared import BridgeClient
from soku_rl.env.encoding import AGENTS
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.vector_env import TwoPlayerVectorEnv
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from soku_rl.env.v3c import V3RewardConfig, V3Ledger, vocabulary
from soku_rl.marl.br import OpponentEntry
from soku_rl.policy.loader import load_policy
from soku_rl.policy.matchups import opponent_interface
from soku_rl.rl import configure_runtime, ppo_settings
from soku_rl.rl.command_ppo import CommandMatchupEnv
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.combo_telemetry import ComboTelemetry

ROOT = Path(os.environ['SOKURL_RL_ROOT'])


def configuration(arm):
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'config'), version_base='1.3'):
        cfg = OmegaConf.to_container(hydra.compose(config_name='v3c_comparison'), resolve=True)
    cfg['comparison']['arm'] = arm
    if arm == 'V3C':
        cfg['episode']['combat_reward']['combo_beta'] = 0.
        cfg['rl']['ppo']['mask_card_actions'] = False
    cfg['algorithm']['ppo'] = deepcopy(cfg['rl']['ppo'])
    ppo_settings(cfg)
    return cfg


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')


class NativeStates:
    def __init__(self, backend, stack):
        self.backend, self.stack, self.clients = backend, stack, {}

    def __call__(self, slot, seat):
        if slot not in self.clients:
            pid = self.backend.request('status', [slot])[slot]['pid']
            self.clients[slot] = self.stack.enter_context(BridgeClient(pid))
        snapshot = self.clients[slot].snapshot()
        if snapshot.run_state_name != 'PAUSED' or snapshot.latest.frameId != snapshot.game_frame:
            raise RuntimeError('native command state must be the exact paused frame')
        player = (snapshot.latest.p1, snapshot.latest.p2)[seat]
        return dict(frame=int(snapshot.game_frame), segment=int(snapshot.latest.segmentId),
            action=int(player.actionId), frame_flags=int(player.frameFlags),
            facing=int(player.facing), x=float(player.x), y=float(player.y),
            hitstop=int(player.hitstop), untech=int(player.untech))


def make_view(config, output, stack, budget):
    backend = stack.enter_context(closing(WorkerBackend(log_path=output / 'worker.log', **config['runtime'])))
    episode = EpisodeConfig.from_dict(config['episode'])
    backend.configure_observation(episode.backend_observation())
    env = LearningVectorEnv(TwoPlayerVectorEnv(backend, 2, episode), LearningConfig(**config['wrappers']))
    stack.callback(env.close)
    population = config['algorithm']['opponents']
    opponents = [OpponentEntry(entry['name'], load_policy(entry['name'], entry['policy'],
        opponent_interface(env.interface, config['algorithm']['matchups']['learner'], entry), config['device']))
        for entry in population]
    view = CommandMatchupEnv(env, 'balanced', opponents, [1.], config['seed'],
        config['algorithm']['matchups']['learner'], [e['setup'] for e in population],
        native_budget=budget, gamma=config['rl']['ppo']['gamma'],
        state_reader=NativeStates(backend, stack), v3=config['comparison']['arm'] == 'V3C')
    reward = V3RewardConfig(**{k: config['comparison'][k] for k in asdict(V3RewardConfig())})
    # Explicit experiment configuration, not an implicit global reward default.
    view.ledger_factory = lambda: V3Ledger(reward)
    stack.callback(view.close)
    return view, backend


class Records(BaseCallback):
    def __init__(self, output, view):
        super().__init__()
        self.output, self.view = output, view
        self.rollouts, self.episodes_written = [], 0
        self.combo = ComboTelemetry()
        self.parts = {}
        self.commands = {}
        self.next_checkpoint = 16384
        self.decisions = 0
        self.accepted = self.interrupted = 0
        self.censored_frames = 0

    def _on_rollout_start(self):
        self.started = time.perf_counter()
        self.combo.reset_batch()
        self.parts = {}
        if not all(torch.isfinite(p).all() for p in self.model.policy.parameters()):
            raise FloatingPointError('nonfinite policy parameters')

    def _on_step(self):
        for record in self.view.frame_records:
            for key, value in record['parts'].items():
                self.parts[key] = self.parts.get(key, 0.) + value
            info = dict(native_combat_events=record['events'], learner_seat=record['learner_seat'],
                        combat_reward_summary=record['combat'],
                        episode_id=(record['episode']['world_seed'], record['episode']['opponent_seed']))
            self.combo.observe(record['slot'], info, record['done'], self.num_timesteps)
        self.view.frame_records.clear()
        for info in self.locals['infos']:
            if not info['valid_decision']:
                continue
            self.decisions += 1
            command = str(info['command'])
            self.commands[command] = self.commands.get(command, 0) + 1
            feedback = info.get('command_feedback', {})
            self.accepted += int(feedback.get('accepted', False))
            self.interrupted += int(feedback.get('interrupted', False))
            if feedback.get('budget_cut'):
                self.censored_frames += info['duration']
        self.flush_episodes()
        if int(self.view.native_frames.sum()) >= self.next_checkpoint:
            self.model.save(self.output / f'checkpoint-{self.next_checkpoint}.zip')
            self.next_checkpoint += 16384
        return True

    def flush_episodes(self):
        with (self.output / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
            for row in self.view.episode_stats[self.episodes_written:]:
                stream.write(json.dumps(row) + '\n')
        self.episodes_written = len(self.view.episode_stats)

    def _on_rollout_end(self):
        row = dict(native_frames=self.view.native_frames.tolist(), decisions=self.view.decisions.tolist(),
            rollout=len(self.rollouts) + 1, seconds=time.perf_counter() - self.started,
            reward_hp=self.parts.copy(), combo=self.combo.snapshot(),
            ppo_updates=self.model._n_updates)
        self.rollouts.append(row)
        with (self.output / 'rollouts.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        write_json(self.output / 'progress.json', dict(rollouts=len(self.rollouts),
            native_frames=self.view.native_frames.tolist(), decisions=self.view.decisions.tolist(),
            episodes=self.episodes_written, commands=self.commands))


def validate_commands(view, output, only=None):
    """Known community command IDs, both native seats; no mechanics estimation."""
    class Neutral:
        def act(self, _):
            return view.env.interface.action(256)
    expected = {'7': {8}, '8': {6}, '9': {7}, '7D': {210}, '8D': {208}, '9D': {209},
                '6D': {200}, '44': {201}, '66': {200}, '5A': {300,301}, '2A': {303},
                '3A': {304}, '6A': {302}, '5B': {400}, '2B': {402}, '6B': {401},
                '5C': {410}, '2C': {412}, '6C': {411}, '236B': {560}, '236C': {561},
                '623B': {520}, '623C': {521}, '214B': {500}, '214C': {500},
                '22B': {540}, '22C': {541}, '2B{40}': {402}, '5C{40}': {410},
                'forward': {4}, 'back_guard': {5}, 'crouch_guard': {1,2}, 'neutral': set()}
    cases = []
    view.reset()
    for index, (name, program) in enumerate(view.programs):
        if only is not None and name != only:
            continue
        view._reset_slots({0:1732 + index * 2, 1:1733 + index * 2})
        view.actors = {0: Neutral(), 1: Neutral()}
        for _ in range(600):
            view.step(np.zeros(2,dtype=int))
            view.frame_records.clear()
        _, _, _, infos = view.step(np.full(2, index))
        records = list(view.frame_records)
        view.frame_records.clear()
        for _ in range(8):
            view.step(np.zeros(2,dtype=int))
            records.extend(view.frame_records)
            view.frame_records.clear()
        actions = {s: {e['action_after'] for row in records if row['slot'] == s
                      for e in row['events'] if e['kind'] == 'ACTION_TRANSITION'
                      and e['attacker_seat'] == view.players[s] and e.get('accepted')}
                   for s in (0,1)}
        call_actions = {str(s): sorted(a) for s,a in actions.items()}
        if index < 13:
            for row in records:
                actions[row['slot']].add(row['after'][view.players[row['slot']]]['action'])
        case = dict(command=name, expected=sorted(expected[name]),
                    set_action_calls=call_actions,
                    evidence='NATIVE_FRAME_BOUNDARY_ACTION' if index < 13 else 'NATIVE_SET_ACTION_CALL',
                    observed={str(s):sorted(a) for s,a in actions.items()},
                    durations=[i['duration'] for i in infos],
                    expected_duration=sum(stage.frames for stage in program),
                    status='PASS' if all(not expected[name] or bool(a & expected[name]) for a in actions.values()) else 'FAIL')
        case['samples'] = records[:6]
        if any(i['duration'] != case['expected_duration'] for i in infos):
            case['status'] = 'FAIL'
        cases.append(case)
        write_json(output / 'command-validation.json', cases)
    if any(c['status'] != 'PASS' for c in cases):
        raise RuntimeError('one or more candidate commands failed native validation; training blocked')
    return cases


def validate_rewards(view, output):
    class Neutral:
        def act(self, _):
            return view.env.interface.action(256)
    indices = {name: n for n,(name,_) in enumerate(view.programs)}
    view._reset_slots({0:88231,1:88232})
    view.actors = {0:Neutral(),1:Neutral()}
    for _ in range(600):
        view.step(np.zeros(2,dtype=int)); view.frame_records.clear()
    checks = []
    for _ in range(3):
        view.step(np.full(2,indices['7'])); view.frame_records.clear()
        for wait in range(240):
            states = [view.state_reader(s,view.players[s]) for s in (0,1)]
            if all(s['action'] == 0 and s['y'] == 0 for s in states):
                break
            view.step(np.zeros(2,dtype=int)); view.frame_records.clear()
        else:
            raise RuntimeError('back-jump validation did not recover in bounded time')
    for s in (0,1):
        row = view.ledgers[s].totals[view.players[s]]
        checks.append(dict(slot=s,back_jumps=row['back_jumps'],cost=row['back_jump_cost'],
                           status='PASS' if row['back_jumps']==3 and row['back_jump_cost']==120 else 'FAIL'))
    view.step(np.full(2,indices['44'])); view.frame_records.clear()
    for _ in range(60):
        view.step(np.zeros(2,dtype=int)); view.frame_records.clear()
    for s in (0,1):
        row = view.ledgers[s].totals[view.players[s]]
        checks.append(dict(slot=s,back_d=row['back_d'],cost=row['back_d_cost'],
                           status='PASS' if row['back_d']==1 and row['back_d_cost']==20 else 'FAIL'))
    for _ in range(160):
        states = [view.state_reader(s,view.players[s]) for s in (0,1)]
        distances = [abs(view.state_reader(s,0)['x']-view.state_reader(s,1)['x']) for s in (0,1)]
        if all(d < 45 for d in distances):
            break
        view.step(np.array([indices['forward'] if d>=45 else 0 for d in distances]))
        view.frame_records.clear()
    hits = []
    for _ in range(320):
        view.step(np.full(2,indices['5A']))
        for row in view.frame_records:
            relevant = [e for e in row['events'] if e['kind']=='HIT_RESOLUTION'
                        and e['attacker_seat']==row['learner_seat'] and e['hit_or_block']=='HIT'
                        and e['hp_before']>e['hp_after']]
            for event in relevant:
                hits.append(dict(slot=row['slot'],frame=row['frame'],event=event,parts=row['parts']))
        view.frame_records.clear()
        if all(any(h['slot']==s and h['event']['combo_count_after']>=3 for h in hits) for s in (0,1)):
            break
    for s in (0,1):
        checks.append(dict(slot=s,native_hit_ordinals=[h['event']['combo_count_after'] for h in hits if h['slot']==s],
            status='PASS' if any(h['slot']==s and h['event']['combo_count_after']>=3 for h in hits) else 'FAIL'))
    for h in hits:
        e = h['event']; n = e['combo_count_after']
        expected = (e['hp_before']-e['hp_after']) * ((.5 if n==1 else 1+.2*(n-2))-1)
        if abs(h['parts']['combo_hit_scaling']-expected)>1e-6 or h['parts']['combo'] != 0:
            raise RuntimeError('native combo event reward disagrees with V3-C definition')
    write_json(output/'native-reward-validation.json',dict(checks=checks,hits=hits))
    if any(c['status']!='PASS' for c in checks):
        raise RuntimeError('native avoidance/combo validation failed; training blocked')
    return checks


def evaluate(model, view, config, output):
    view.auto_reset = False
    combo = ComboTelemetry()
    for index, seed in enumerate(config['comparison']['evaluation_seeds']):
        view.rng = np.random.default_rng(config['comparison']['evaluation_policy_seed'] + index)
        torch.manual_seed(config['comparison']['evaluation_policy_seed'] + index)
        view._reset_slots({0:seed,1:seed})
        while len(view.completed_slots) < 2:
            if view.finished:
                raise RuntimeError('evaluation budget exhausted before paired episodes completed')
            actions, _ = model.predict(view._stack(), deterministic=False)
            _, rewards, dones, infos = view.step(actions)
            for row in view.frame_records:
                info = dict(native_combat_events=row['events'],learner_seat=row['learner_seat'],
                            episode_id=(seed,row['slot']),combat_reward_summary=row['combat'])
                combo.observe(row['slot'],info,row['done'])
            view.frame_records.clear()
        write_json(output/'evaluation-games.json',view.episode_stats)
    rows = view.episode_stats
    by_seat = {}
    for seat in (0,1):
        games = [r for r in rows if r['context']['player']==seat]
        by_seat[str(seat)] = dict(games=len(games),wins=sum(r['outcome']==f'p{seat+1}_win' for r in games),
            timeouts=sum(r['outcome']=='time_limit' for r in games),
            mean_native_frames=float(np.mean([r['l'] for r in games])),
            mean_damage_balance=float(np.mean([r['combat']['damage_dealt']-r['combat']['damage_taken'] for r in games])))
    summary = dict(by_seat=by_seat,combo=combo.snapshot(),
        wins=sum(r['outcome']==f"p{r['context']['player']+1}_win" for r in rows),
        games=len(rows),native_frames=view.native_frames.tolist(),decisions=view.decisions.tolist(),
        interaction=view.interaction,
        limitations=['16 paired games are a screening sample, not significance evidence.',
                     'Different action vocabularies/durations prevent identical policy RNG consumption.',
                     'Free-flight214 is not classified as back D;201/203 are native-confirmed.',
                     'Approach/close-range counters are analysis only; no distance reward.'])
    write_json(output/'evaluation.json',summary)
    return summary


def run(args):
    config = configuration(args.arm)
    budget = args.frames or config['comparison']['native_frames']
    if budget > config['comparison']['native_frames'] or budget < 2 or budget % 2:
        raise ValueError('bounded even native-frame budget required')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('preserve existing output')
    output.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(config), output / 'config.yaml')
    configure_runtime(config['rl'])
    write_json(output / 'protocol.json', dict(budget=budget, per_slot=budget // 2,
        seed=config['seed'], arm=args.arm, initialization='FRESH_PAIRED_SEED_NO_OLD_CHECKPOINT',
        action_contract='MARISA_SINGLE_COMMAND_V1' if args.arm == 'V3C' else 'V2_KEYS_90',
        vocabulary=[(n, [asdict(s) for s in p]) for n, p in vocabulary()],
        evaluation_seeds=config['comparison']['evaluation_seeds'],
        ppo=config['rl']['ppo'], timestamp=datetime.now().isoformat(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [*sorted((ROOT/'src').rglob('*.py')), Path(__file__)]}))
    if args.stage == 'plan':
        print(json.dumps(dict(budget=budget, config=config['rl']['ppo'], output=str(output)), indent=2))
        return
    started = time.perf_counter()
    report = dict(status='FAIL', stage=args.stage)
    try:
        with ExitStack() as stack:
            view, backend = make_view(config, output, stack, budget)
            model, _ = create_ppo(view, view.env.interface, config['algorithm'], {'kind': 'fresh'},
                                  config['device'], config['seed'])
            if args.stage == 'evaluate':
                if args.model is None:
                    raise ValueError('evaluation requires an explicit model')
                from soku_rl.rl.command_ppo import DurationPPO
                protocol = json.loads((args.model.parent/'protocol.json').read_text())
                expected = 'MARISA_SINGLE_COMMAND_V1' if args.arm=='V3C' else 'V2_KEYS_90'
                if protocol['action_contract'] != expected:
                    raise ValueError('checkpoint action contract mismatch')
                model = DurationPPO.load(args.model,env=view,device=config['device'])
                report['evaluation_model_sha256'] = hashlib.sha256(args.model.read_bytes()).hexdigest()
            if args.paired_initial:
                from stable_baselines3 import PPO
                source = PPO.load(args.paired_initial, device=config['device'])
                previous = source.policy.state_dict()
                state = model.policy.state_dict()
                for key in state:
                    if not key.startswith('action_net.'):
                        if key not in previous or state[key].shape != previous[key].shape:
                            raise ValueError(f'paired initialization mismatch: {key}')
                        state[key] = previous[key]
                model.policy.load_state_dict(state, strict=True)
                report['paired_initial_sha256'] = hashlib.sha256(args.paired_initial.read_bytes()).hexdigest()
            model.set_logger(configure(str(output / 'scalars'), ['csv']))
            model.save(output / 'initial.zip')
            report['initial_hash'] = parameter_hash(model.policy)
            report['runtime'] = backend.identity
            records = Records(output, view)
            optimizer_steps = [0]
            def updated(*_):
                optimizer_steps[0] += 1
            step_hook = model.policy.optimizer.register_step_post_hook(updated)
            stack.callback(step_hook.remove)
            if args.stage == 'evaluate':
                report['evaluation'] = evaluate(model,view,config,output)
            elif args.stage == 'validate':
                if args.arm != 'V3C':
                    raise ValueError('command validation requires V3C')
                if not args.rewards_only:
                    report['command_validation'] = validate_commands(view, output, args.only_command)
                if args.only_command is None:
                    report['reward_validation'] = validate_rewards(view,output)
            elif args.stage == 'train':
                model.learn(total_timesteps=budget, callback=records)
                model.save(output / 'final.zip')
                report.update(final_hash=parameter_hash(model.policy), ppo_updates=model._n_updates,
                              optimizer_steps=optimizer_steps[0],
                              censored_native_frames_excluded_from_ppo=records.censored_frames,
                              rollouts=len(records.rollouts), commands=records.commands,
                              accepted_command_decisions=records.accepted,
                              interrupted_command_decisions=records.interrupted)
                report['last_ppo_metrics'] = {k:v for k,v in model.logger.name_to_value.items() if k.startswith('train/')}
            else:
                view.seed(config['seed'])
                view.reset()
                accepted = {}
                observations = []
                for n in range(400):
                    action = (n // 10) % view.action_space.n
                    _, rewards, dones, infos = view.step(np.full(2, action))
                    if not np.isfinite(rewards).all():
                        raise FloatingPointError('nonfinite smoke reward')
                    for record in view.frame_records:
                        for event in record['events']:
                            if event['kind'] == 'ACTION_TRANSITION' and event.get('accepted'):
                                key = str(event['attacker_seat'])
                                accepted.setdefault(key, set()).add(event['action_after'])
                    observations.append(infos)
                    view.frame_records.clear()
                    if view.finished:
                        break
                write_json(output / 'smoke.json', dict(infos=observations, accepted={k: sorted(v) for k,v in accepted.items()}))
                report['accepted_actions'] = {k: sorted(v) for k, v in accepted.items()}
            report.update(status='PASS', native_frames=view.native_frames.tolist(),
                decisions=view.decisions.tolist(), episodes=view.episode_stats,
                active_partial_episode_frames=view.lengths.tolist())
            report['partial_episode_ledgers'] = {str(s): view.ledgers[s].totals[view.players[s]]
                                                for s in view.ledgers}
            report['interaction'] = view.interaction
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['seconds'] = time.perf_counter() - started
        write_json(output / 'result.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','runtime')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'validate', 'smoke', 'train', 'evaluate'), default='plan')
    parser.add_argument('--arm', choices=('V2', 'V3C'), required=True)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paired-initial', type=Path)
    parser.add_argument('--only-command')
    parser.add_argument('--rewards-only', action='store_true')
    parser.add_argument('--model', type=Path)
    run(parser.parse_args())

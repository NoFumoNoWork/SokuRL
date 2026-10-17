"""Bounded frozen-policy native duel, reusing existing event/primitive executors."""
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from compare_v3c import NativeStates
from soku_rl.env.encoding import AGENTS
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.env.vector_env import TwoPlayerVectorEnv
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from soku_rl.policy.hierarchical_bc import HierarchicalEpisode
from soku_rl.policy.hierarchical_control import candidate_opportunity
from soku_rl.policy.loader import load_policy
from soku_rl.replay.human_dataset import sha256, save_json
from soku_rl.rl.factorized_bc import load_event_actor


def duel_result(outcome, bc_seat, hp):
    if bc_seat not in (0, 1) or len(hp) != 2:
        raise ValueError('two seats and final HP required')
    if outcome in ('p1_win', 'p2_win'):
        return 'BC_KO_WIN' if outcome == f'p{bc_seat+1}_win' else 'PPO_KO_WIN'
    if outcome == 'double_ko':
        return 'DOUBLE_KO'
    if outcome == 'time_limit':
        difference = hp[bc_seat] - hp[1-bc_seat]
        return 'TIMEOUT_BC_HP_LEAD' if difference > 0 else 'TIMEOUT_PPO_HP_LEAD' if difference < 0 else 'TIMEOUT_HP_TIE'
    raise ValueError('only native terminal or explicit horizon may be scored')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bc', type=Path, required=True)
    p.add_argument('--ppo-directory', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[81730011, 81730021, 81730031, 81730041])
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    if (out.exists() or not out.is_relative_to(root/'rl/runs') or not 1 <= len(args.seeds) <= 4
            or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 0xffffffff for s in args.seeds)):
        raise ValueError('new local output and at most four paired world seeds required')
    config = OmegaConf.to_container(OmegaConf.load(args.ppo_directory/'config.yaml'), resolve=True)
    episode = EpisodeConfig.from_dict(config['episode'])
    if (episode.max_frames, episode.decision_frames, episode.latency_frames) != (7200, 1, 0):
        raise ValueError('preserve7200F/1F/zero-latency contract')
    if config['algorithm']['matchups']['learner']['character'] != 1:
        raise ValueError('both frozen policies must retain own trained Marisa character')
    episode = replace(episode, match=MatchConfig(PlayerSetup(1, 0, 0), PlayerSetup(1, 0, 0)))
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    report = dict(status='RUNNING', games=[], seeds=args.seeds, matchup='MARISA_MIRROR',
        rl_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        rl_dirty=subprocess.check_output(['git', 'status', '--short', '--', 'rl'], cwd=root, text=True).strip(),
        platform_reference=subprocess.check_output(['git', 'rev-parse', 'origin/platform'], cwd=root, text=True).strip(),
        source_identity='Wrapper before.json/after.json bind actual source hashes;not assumed clean Platform.',
        bc=str(args.bc.resolve()), bc_sha256=sha256(args.bc),
        ppo=str((args.ppo_directory/'final.zip').resolve()), ppo_sha256=sha256(args.ppo_directory/'final.zip'),
        ppo_config_sha256=sha256(args.ppo_directory/'config.yaml'),
        environment=dict(clear=True, cards_disabled=True, horizon=7200, decision_frames=1, slots=2),
        inference=dict(bc='EXISTING_CATEGORY_FIRST_GREEDY_EVENT_CONTROLLER',
                       ppo='EXISTING_SEEDED_STOCHASTIC_PPO_EPISODE', max_wait=60),
        limitations=['Mirror opponent is outside both training matchups.',
            'BC inherited PPO feature weights;not compute-matched training-method comparison.',
            'Control candidate masks are NOT complete engine-legal action sets.',
            'Timeout HP leads are NOT native KO wins.', 'Small paired screen,not significance evidence.'],
        training='NOT RUN', native_changes=False, progress={})
    save_json(out/'result.json', report)
    started = time.perf_counter()
    try:
        actor, metadata = load_event_actor(args.bc, 'cpu')
        catalog = metadata['catalog']
        allowed = [c['name'] for c in catalog]
        report['allowed_candidates'] = allowed
        with ExitStack() as stack:
            backend = WorkerBackend([sys.executable, '-B', str(root/'rl/tools/rollout_worker.py')], str(root),
                str(out/'worker.log'), 120, 45, True, str(root/'th123_jp'), fresh_reset=True,
                cards_disabled=True, fixed_weather='clear', combat_events=True, native_input_counters=True)
            stack.callback(backend.close)
            backend.configure_observation(episode.backend_observation())
            env = LearningVectorEnv(TwoPlayerVectorEnv(backend, 2, episode), LearningConfig(**config['wrappers']))
            stack.callback(env.close)
            reader = NativeStates(backend, stack)
            ppo = load_policy('direct-ppo', dict(kind='sb3', path=report['ppo'],
                training_config=str((args.ppo_directory/'config.yaml').resolve())), env.interface, 'cpu')
            for seed in args.seeds:
                observations, _ = env.reset_matchups({0: seed, 1: seed}, {0: episode.match, 1: episode.match})
                controllers = {s: HierarchicalEpisode(actor, catalog, env.interface, 'cpu', 60,
                    'masked_native_feedback') for s in (0, 1)}
                opponents = {s: ppo.spawn(seed ^ 0x51A7) for s in (0, 1)}
                events, inputs = {0: [], 1: []}, {0: [], 1: []}
                initial_hp = {s: list(env.episodes[s].reward.hp) for s in (0, 1)}
                active = {0, 1}
                while active:
                    joint = {}
                    for s in sorted(active):
                        frame = env.episodes[s].frame
                        state = reader(s, s)
                        if state['frame'] != frame:
                            raise RuntimeError('control observation and native frame differ')
                        bc, _ = controllers[s].act(observations[s][AGENTS[s]], frame=frame,
                            facing=state['facing'], segment=state['segment'], events=events[s], seat=s,
                            opportunity=candidate_opportunity(catalog, state, allowed))
                        other = opponents[s].act(observations[s][AGENTS[1-s]])
                        joint[s] = {AGENTS[s]: bc, AGENTS[1-s]: other}
                        inputs[s].append([joint[s][a] for a in AGENTS])
                    observations, _, terms, truncs, infos = env.step(joint)
                    for s in sorted(active):
                        events[s] = infos[s][AGENTS[s]]['native_combat_events']
                        if not (any(terms[s].values()) or any(truncs[s].values())):
                            continue
                        controller = controllers[s]
                        controller.controller.observe_feedback(events[s], s)
                        controller.controller.finish_attempt('NATIVE_EPISODE_END')
                        hp = list(env.episodes[s].reward.hp)
                        outcome = infos[s][AGENTS[s]]['outcome']
                        arrays = out/f'{seed}-bc-seat{s}-inputs.npz'
                        np.savez_compressed(arrays, primitive_actions=np.asarray(inputs[s], np.int16))
                        game = dict(seed=seed, bc_seat=s, frames=env.episodes[s].frame,
                            outcome=outcome, result=duel_result(outcome, s, hp), initial_hp=initial_hp[s], final_hp=hp,
                            bc_hp_difference=hp[s]-hp[1-s], bc_decisions=controller.decisions,
                            bc_commands=dict(Counter(a['command'] for a in controller.controller.attempts)),
                            bc_acceptance=dict(Counter(a['acceptance'] for a in controller.controller.attempts)),
                            reward_summary=[infos[s][a]['combat_reward_summary'] for a in AGENTS],
                            input_path=arrays.name, input_sha256=sha256(arrays))
                        report['games'].append(game)
                        report['counts'] = dict(Counter(g['result'] for g in report['games']))
                        save_json(out/'result.json', report)
                        print(json.dumps(game), flush=True)
                    active = {s for s in active if not env.episodes[s].ended}
                    if active and max(env.episodes[s].frame for s in active) % 600 == 0:
                        report['progress'] = dict(seed=seed, frames={s: env.episodes[s].frame for s in active},
                            seconds=time.perf_counter()-started)
                        save_json(out/'result.json', report)
                        print(json.dumps(report['progress']), flush=True)
            if len(report['games']) != 2*len(args.seeds):
                raise RuntimeError('paired games incomplete')
        report.update(status='PASS_PAIRED_NATIVE_DUEL', mean_bc_hp_difference=float(np.mean(
            [g['bc_hp_difference'] for g in report['games']])))
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        report['seconds'] = time.perf_counter()-started
        save_json(out/'result.json', report)


if __name__ == '__main__':
    main()

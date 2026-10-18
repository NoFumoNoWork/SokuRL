"""Static source audit plus a declared synthetic observation compatibility probe.

No game process, native reads, student training or new gameplay measurement.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import numpy as np

from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.env.observation.animation import frame_width
from soku_rl.policy.god.query import QueryTeacher


def source_audit(paths):
    groups = {
        'privileged_or_unavailable': r'get_(special_data|key_stat2|key_map2|obj_attackarea|obj_hitarea|attackarea|hitarea|fflags|aflags|correction|skill_lv)|enemy_(keys|frame|xspeed|yspeed)',
        'home_advantage': r'stage_number|bgm_number|after_guard_count',
        'randomness': r'math\.random|randomseed',
        'inputs_and_timing': r'key_(on|off|reset|event)|set_(key|data|weather)_delay|\bwait\(|\byield\(',
        'stateful_decision': r'create_thread|function (main|calculation|base_states)',
    }
    result = []
    for path in paths:
        data = path.read_bytes()
        lines = data.decode('cp932', 'replace').splitlines()
        evidence = {group: [{'line': n, 'code': line.encode('ascii', 'backslashreplace').decode()}
            for n,line in enumerate(lines,1) if re.search(pattern,line)] for group,pattern in groups.items()}
        result.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), evidence=evidence))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-student', action='store_true',
                        help='verify exact existing PPO weight initialization on CPU, no environment stepping')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    if not output.is_relative_to(root/'rl/runs'):
        raise ValueError('audit output must be under this workspace rl/runs')
    output.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(config_dir=str(root/'rl/config'), version_base='1.3'):
        config = OmegaConf.to_container(compose(config_name='train_combat_kickstarting'), resolve=True)
    interface = LearningInterface(EpisodeConfig.from_dict(config['episode']), LearningConfig(**config['wrappers']))
    spec = config['rl']['kickstarting']['teacher']
    teacher = QueryTeacher(interface, spec)
    package = Path(spec['package'])
    sources = [package/spec['script'], package/'00_reimu_main.ai', package/'def_clc.lua',
               package/'function_command.txt', Path(spec['api_source'])]
    probe = []
    # Explicit fixture, not recorded original states or live teacher coverage.
    for character, script in ((1,'01_marisa_main.ai'), (0,'00_reimu_main.ai')):
        variant = QueryTeacher(interface, spec | dict(script=script))
        actor = variant.spawn(spec['seed'])
        width = frame_width(interface.episode.animation_object_slots)
        row = np.zeros(width, np.float32)
        row[0] = row[10] = 1
        row[3], row[13], row[4], row[14] = 1, -1, character, 1-character
        obs = np.r_[np.tile(row, interface.episode.history_frames),
                    np.zeros(8*interface.config.action_history, np.float32)]
        statuses = Counter()
        for _ in range(spec['reaction_frames']+2):
            answer = actor.query(obs)
            statuses[answer.status] += 1
        probe.append(dict(script=script, statuses=dict(statuses), last=answer.record()))
    result = dict(status='BLOCKED_TEACHER_ADAPTATION' if any(
        p['last']['status'] != 'VALID_INPUT' for p in probe) else 'PASS_FIXTURE_ONLY',
        evidence_level='SYNTHETIC_CONTRACT_PROBE_NOT_GAMEPLAY_COVERAGE',
        config=config, teacher_fingerprint=teacher.fingerprint, source_audit=source_audit(sources),
        probes=probe, versions={name:version(name) for name in ('lupa','torch','stable-baselines3')},
        rl_head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        dirty=subprocess.check_output(['git','status','--porcelain'], text=True).splitlines(),
        boundaries=dict(control='student only; advice never submitted',
            reaction='entire teacher animation view delayed by configured frames',
            legality='input vocabulary/card mask only, NOT engine acceptance/cancel proof',
            time='one query step per 1F observation; episode-local coroutine and seeded RNG',
            coverage='no real student/native rollout tested in this audit',
            next='adapt source-backed teacher predicates to student-observable semantics; no fake world coordinates'))
    if args.check_student:
        from stable_baselines3 import PPO
        import gymnasium as gym
        from soku_rl.rl.ppo import create_ppo, parameter_hash
        env = gym.Env()
        env.observation_space, env.action_space = interface.observation_space, interface.action_space
        source = config['algorithm']['initial_policy']
        model, metadata = create_ppo(env, interface, config['rl'], source, 'cpu', config['seed'])
        original = PPO.load(source['path'], device='cpu')
        before, after = parameter_hash(original.policy), parameter_hash(model.policy)
        if before != after:
            raise RuntimeError('existing PPO weight initialization differs')
        result['student'] = dict(status='PASS_EXACT_EXISTING_PPO_WEIGHTS', source=metadata,
            original_parameters=before, initialized_parameters=after,
            optimizer='fresh PPO Adam, no teacher-specific optimizer',
            native_steps=0, gradient_updates=0)
    (output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], report=str(output/'report.json'), probes=probe), indent=2))


if __name__ == '__main__':
    main()

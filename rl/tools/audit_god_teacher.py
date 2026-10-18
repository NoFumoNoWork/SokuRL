"""Static source audit plus a declared synthetic observation compatibility probe.

No game process, native reads, student training or new gameplay measurement.
"""
import argparse
import copy
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


def field_dependencies(paths, has_world):
    """Conservative textual references, NOT proof a field is read on every path."""
    from sokurl_platform.native_layout import FIGHTER_NAMES,WORLD_NAMES
    available = {'char','act','act_block','dir'}
    if has_world:
        available |= {'x','y'}
    result = []
    for prefix in ('my_','enemy_'):
        for field in FIGHTER_NAMES:
            name = prefix+field
            references = []
            for path in paths:
                for line,text in enumerate(path.read_bytes().decode('cp932','replace').splitlines(),1):
                    if re.search(r'\b'+re.escape(name)+r'\b',text):
                        references.append(dict(path=str(path),line=line))
            if references:
                status = 'AVAILABLE_SHARED' if field in available else 'NOT_IN_STUDENT_CONTRACT'
                if field == 'hp':status = 'AVAILABLE_QUANTIZED_GAUGE_NOT_EXACT_HP'
                result.append(dict(name=name,status=status,existing_platform='native_layout/native_reader (not enabled for student)',
                    references=references,reference_kind='STATIC_TEXT_OVERAPPROX_INCLUDES_UNEXECUTED_CODE'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-student', action='store_true',
                        help='verify exact existing PPO weight initialization on CPU, no environment stepping')
    parser.add_argument('--world-coordinates',action='store_true')
    parser.add_argument('--student-directory',help='explicit migrated model.zip/config.yaml directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    if not output.is_relative_to(root/'rl/runs'):
        raise ValueError('audit output must be under this workspace rl/runs')
    output.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(config_dir=str(root/'rl/config'), version_base='1.3'):
        config = OmegaConf.to_container(compose(config_name='train_combat_kickstarting'), resolve=True)
    if args.world_coordinates:
        config['episode']['animation_world_coordinates'] = True
        config['rl']['ppo']['policy_kwargs']['features_extractor_kwargs']['world_coordinates'] = True
        if args.check_student and not args.student_directory:
            raise ValueError('world student check requires an explicit migrated --student-directory')
    if args.student_directory:
        source = Path(args.student_directory).resolve(strict=True)
        config['algorithm']['initial_policy'] = dict(kind='weights',path=str(source/'model.zip'),
                                                      training_config=str(source/'config.yaml'))
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
        core_width = frame_width(interface.episode.animation_object_slots)
        width = frame_width(interface.episode.animation_object_slots,interface.episode.animation_world_coordinates)
        row = np.zeros(width, np.float32)
        row[0] = row[10] = 1
        row[3], row[13], row[4], row[14] = 1, -1, character, 1-character
        row[core_width-5:core_width-1] = 1.
        if args.world_coordinates:
            row[core_width:core_width+4] = (320.,0.,800.,0.)
        obs = np.r_[np.tile(row, interface.episode.history_frames),
                    np.zeros(8*interface.config.action_history, np.float32)]
        statuses = Counter()
        for _ in range(spec['reaction_frames']+2):
            answer = actor.query(obs)
            statuses[answer.status] += 1
        probe.append(dict(script=script, statuses=dict(statuses), last=answer.record(),
                          source_adapter=variant.adapter.report if variant.adapter else None))
    result = dict(status='BLOCKED_TEACHER_ADAPTATION' if any(
        p['last']['status'] != 'VALID_INPUT' for p in probe) else 'PASS_FIXTURE_ONLY',
        evidence_level='SYNTHETIC_CONTRACT_PROBE_NOT_GAMEPLAY_COVERAGE',
        config=config, teacher_fingerprint=teacher.fingerprint, source_audit=source_audit(sources),
        field_dependencies=field_dependencies(sources,args.world_coordinates),
        probes=probe, versions={name:version(name) for name in ('lupa','torch','stable-baselines3')},
        rl_head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        dirty=subprocess.check_output(['git','status','--porcelain'], text=True).splitlines(),
        boundaries=dict(control='student only; advice never submitted',
            reaction='entire teacher animation view delayed by configured frames',
            legality='input vocabulary/card mask only, NOT engine acceptance/cancel proof',
            time='one delayed perception tick per 1F observation; disposable intent coroutine only at trusted boundary',
            coverage='no real student/native rollout tested in this audit',
            next='adapt remaining source dependencies without inventing hidden state'))
    result['dependency_inventory'] = [
        dict(feature=name,status=status,existing_source=source,needed=needed)
        for name,status,source,needed in (
            ('screen positions','IMPLEMENTED','existing animation tokens','none'),
            ('world positions','IMPLEMENTED_OPT_IN','existing RenderSnapshot; same visible slots','explicit student migration; no Hook'),
            ('native speed','NOT_EXPOSED','existing RawFrameState.speedX/speedY','shared student feature authorization; do not substitute position finite differences'),
            ('HP','IMPLEMENTED_QUANTIZED','student HP gauge','not exact native HP'),
            ('action/sequence/facing','IMPLEMENTED','student animation tokens','none'),
            ('world distance/front/back','IMPLEMENTED_OPT_IN','derived shared world coordinates','none'),
            ('native spirit/maxSpirit','NOT_EXPOSED','existing RawFrameState','new shared observation approval; no new Hook needed'),
            ('weather/cards-disabled context','NOT_IMPLEMENTED','existing runtime configuration','bind verified environment constants into teacher contract'),
            ('card/skill state','NOT_EXPOSED','existing Platform privileged reader','restricted teacher branch adaptation OR separately approved shared fields'),
            ('object get_obj_data full tuple','PARTIAL_DATA_ONLY','visible object tokens/world positions','do not invent native HP/stable pointer/box counts; adapt API/script'),
            ('boxes/flags/special timers','DISALLOWED_CURRENT_CONTRACT','existing Platform privileged reader','permission/observability decision before any new adapter'),
            ('enemy hidden inputs','PROHIBITED','no allowed student source','remove teacher dependency; do not add a reader'),
            ('independent RNG/delay/query sandbox','IMPLEMENTED','existing ScriptAPI LCG + restricted query','native equivalence not claimed'),
            ('accepted move/cancel legality','NOT_IMPLEMENTED','existing diagnostic action events; no complete allowed legal mask','observable legality contract + bounded native validation'),
            ('teacher intent lifecycle','OFFLINE_VERIFIED','query/intents.py + pinned source_adapter.py',
             'native boundary and 90-action lowering still absent; unsupported segments mask, no old coroutine resumes'),
            ('original decision/perception split','PINNED_SOURCE_ADAPTATION','2011 Marisa/Reimu/main+def_clc hashes',
             'perception dependencies must be shared or explicitly adapted; decision writes rollback, not original policy equivalence'),
            ('joint loss/masks/decay/save','OFFLINE_VERIFIED','existing PPO + opt-in KickstartingPPO','native rollout/coverage verification still required'),
            ('native teacher rollout/training','NOT_TESTED','existing Worker/Bridge infrastructure','unblock teacher labels first; no new deployment system needed'),
        )]
    if args.check_student:
        from stable_baselines3 import PPO
        import gymnasium as gym
        from soku_rl.rl.ppo import create_ppo, parameter_hash
        env = gym.Env()
        env.observation_space, env.action_space = interface.observation_space, interface.action_space
        source = config['algorithm']['initial_policy']
        # Initialization audit is not a live teacher/PPO attachment. Preserve the
        # explicit attachment BLOCKED guard rather than bypassing it in training.
        initialization = copy.deepcopy(config['rl'])
        initialization.pop('kickstarting', None)
        model, metadata = create_ppo(env, interface, initialization, source, 'cpu', config['seed'])
        original = PPO.load(source['path'], device='cpu')
        before, after = parameter_hash(original.policy), parameter_hash(model.policy)
        if before != after:
            raise RuntimeError('existing PPO weight initialization differs')
        result['student'] = dict(status='PASS_EXACT_EXISTING_PPO_WEIGHTS', source=metadata,
            original_parameters=before, initialized_parameters=after,
            optimizer='fresh PPO Adam, no teacher-specific optimizer',
            scope='weight initialization only; no expert attachment or trainability claim',
            native_steps=0, gradient_updates=0)
    (output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], report=str(output/'report.json'), probes=probe), indent=2))


if __name__ == '__main__':
    main()

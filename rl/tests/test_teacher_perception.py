"""The teacher must not gain fields absent from the unchanged student tensor."""
from pathlib import Path

from gymnasium import spaces
import numpy as np
import pytest

from soku_rl.env.observation.animation import frame_width
from soku_rl.env.wrappers.learning import LearningInterface
from soku_rl.policy.god.perception import project, task_context
from soku_rl.policy.god.query import QueryTeacher
from test_kickstarting import interface, spec, API
from test_teacher_intents import boundary


def contract(slots=16):
    old = interface()
    old.episode.animation_world_coordinates = True
    old.episode.animation_object_slots = slots
    old.episode.space = lambda: spaces.Box(-np.inf,np.inf,(8*frame_width(slots,True),),np.float32)
    return LearningInterface(old.episode,old.config)


def row(slots=16, count=1):
    core = frame_width(slots)
    result = np.zeros(frame_width(slots,True),np.float32)
    result[:10] = (1,.25,.5,1,1,400,2,3,4,17)
    result[10:20] = (1,.75,.5,-1,0,9,1,2,3,12)
    result[core-5:core] = (.8,.6,.7,.4,.25)
    result[core:core+4] = (300,0,800,0)
    for i in range(count):
        result[20+10*i:30+10*i] = (1,.5,.5,1,0,800,2,3,4,1)
        result[core+4+2*i:core+6+2*i] = (500+i,100)
    return result


def test_unchanged1832_mapping_and_resource_units():
    values,view = project(row(),16,True,8,6)
    assert contract().observation_space.shape == (8*221+8,)  # test fixture has 1 own input row
    from dataclasses import replace
    full=LearningInterface(contract().episode,replace(contract().config,action_history=8))
    assert full.observation_space.shape == (1832,)
    assert values['my_frame'] == 17 and values['enemy_frame'] == 12
    assert values['my_subsequence'] == 3 and values['my_pose_frame'] == 4
    assert values['my_screen_x'] == 160 and values['my_screen_y'] == 240
    assert values['my_spirit_fraction'] == pytest.approx(.7)
    assert 'my_rei' not in values and 'my_rmax' not in values and 'battle_time' not in values
    assert values['teacher_query_index'] == 8 and values['teacher_observation_index'] == 2
    assert view.objects[0].action == 800 and view.objects[0].owner == 1
    report = view.report()
    assert report['selection_complete'] is True and report['native_list_complete'].startswith('UNKNOWN')
    assert report['enemy'] == 1 and report['own'] == 0


@pytest.mark.parametrize('count,expected', [(0,True),(15,True),(16,None)])
def test_no_fake_complete_native_count_or_truncation_claim(count, expected):
    values,view = project(row(count=count),16,True,8,6)
    assert view.selection_complete is expected
    assert values['visible_enemy_obj_n'] == count
    assert 'enemy_obj_n' not in values
    if count == 16:
        assert values['visible_selection_complete'] is False and values['visible_selection_known'] is False
        assert values['visible_selection_status'] == b'UNKNOWN_AT_CAPACITY'
        assert view.report()['truncation'] == 'POSSIBLE_AT_CAPACITY'


def teacher(tmp_path, source):
    if not API.is_file():pytest.skip('external community API unavailable')
    directory = tmp_path/'script'
    directory.mkdir()
    (directory/'fixture.ai').write_text(source,encoding='ascii')
    return QueryTeacher(contract(), spec(directory) | {'mode':'intent'})


def query(teacher, first, last=None):
    actor = teacher.spawn(17)
    obs = lambda x: np.r_[np.tile(x,8),np.zeros(8,np.float32)]
    actor.query(obs(first))
    actor.query(obs(last if last is not None else first))
    answer = actor.query(obs(last if last is not None else first),opportunity=boundary(),boundary_id=2)
    return actor,answer


def test_delayed_view_and_strict_object_api(tmp_path):
    policy = teacher(tmp_path, '''
function teacher_decide()
 local o=get_visible_obj_data(1,0)
 if o.action==800 and o.x==500 and my_frame==17 and my_spirit_fraction<0.8 then
  command('236B')
 end
end
''')
    changed = row(); changed[9] = 99;changed[frame_width(16)+4] = 700
    actor,answer = query(policy,row(),changed)
    assert answer.intent == '236B' and actor.visible_view.objects[0].x == 500
    assert answer.action is None  # no Phase3 primitive label is manufactured


@pytest.mark.parametrize('expression', ['o.hp','o.address','o.frame','o.attackarea_n','o.img'])
def test_missing_native_object_data_not_filled(tmp_path,expression):
    policy=teacher(tmp_path,f'function teacher_decide() local o=get_visible_obj_data(1,0); local x={expression}; command("5A") end')
    _,answer=query(policy,row())
    assert answer.status=='BLOCKED_INTENT_SOURCE'
    assert 'UNAVAILABLE_VISIBLE_OBJECT_FIELD' in answer.detail
    assert answer.action is None


def test_lua_cannot_modify_public_object_record(tmp_path):
    policy=teacher(tmp_path,'function teacher_decide() local o=get_visible_obj_data(1,0); o.action=0 end')
    _,answer=query(policy,row())
    assert 'READ_ONLY_VISIBLE_OBJECT' in answer.detail


def test_task_declarations_not_native_weather_or_skills(tmp_path):
    context=task_context(dict(fixed_weather='clear',cards_disabled=True))
    assert context['binding'].startswith('CONFIGURATION_ONLY')
    with pytest.raises(ValueError):task_context(dict(fixed_weather='typhoon',cards_disabled=True))
    policy=teacher(tmp_path,"function teacher_decide() if task_cards_disabled then command('5A') end end")
    policy=QueryTeacher(policy.interface,policy.spec | {'task_context':dict(fixed_weather='clear',cards_disabled=True)})
    actor,answer=query(policy,row())
    assert answer.intent=='5A'
    assert actor.lua.eval(b'function(e) return rawget(e,"weather"),rawget(e,"my_rei"),rawget(e,"enemy_obj_n") end')(actor.env)==(None,None,None)


def test_phase1_route_preserves_vm_history_without_legacy_list_reads_or_rollback(tmp_path):
    from dataclasses import replace
    policy=teacher(tmp_path,"function main() error('ORIGINAL_MAIN_MUST_NOT_RUN_IN_PHASE1') end")
    full=LearningInterface(policy.interface.episode,replace(policy.interface.config,action_history=8))
    policy=QueryTeacher(full,policy.spec | {'perception':'student_visible_v1'})
    actors=[policy.spawn(17),policy.spawn(18)]
    obs=np.r_[np.tile(row(count=16),8),np.zeros(64,np.float32)]
    for _ in range(6):
        answer=actors[0].query(obs)
    assert answer.status=='PHASE1_PERCEPTION_READY' and answer.action is None
    assert len(actors[0].perception_history)==4 and not actors[1].perception_history
    assert actors[0].env[b'visible_enemy_obj_n']==16
    assert actors[0].env[b'visible_selection_complete'] is False
    assert actors[0].env[b'visible_selection_known'] is False
    assert actors[0].env[b'visible_selection_status']==b'UNKNOWN_AT_CAPACITY'
    assert actors[0].blocked is None and not actors[0].intents.loaded
    reset=policy.spawn(17)
    assert not reset.perception_history and reset.query(obs).status=='REACTION_WARMUP'


def test_visible_profile_composes_the_actual1832_contract(monkeypatch):
    import sys
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from soku_rl.env.hisouten_env import EpisodeConfig
    from soku_rl.env.wrappers.learning import LearningConfig
    root=Path(__file__).resolve().parents[2]
    for key,value in dict(SOKURL_RL_ROOT=root/'rl',SOKURL_PROJECT_ROOT=root,
        SOKURL_PLATFORM_ROOT=root/'platform',SOKURL_GAME_DIR=root/'th123_jp',
        SOKURL_PYTHON=sys.executable,SOKURL_WORLD_STUDENT=root/'rl/runs/teacher-world-20261018/student').items():
        monkeypatch.setenv(key,str(value))
    with initialize_config_dir(config_dir=str(root/'rl/config'),version_base='1.3'):
        config=OmegaConf.to_container(compose(config_name='train_combat_kickstarting_visible'),resolve=True)
    contract=LearningInterface(EpisodeConfig.from_dict(config['episode']),LearningConfig(**config['wrappers']))
    assert contract.observation_space.shape==(1832,)
    settings=config['rl']['kickstarting']['teacher']
    assert settings['perception']=='student_visible_v1' and settings['mode']=='intent'
    assert settings['task_context']==dict(fixed_weather='clear',cards_disabled=True)

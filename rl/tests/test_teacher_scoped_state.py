"""Scope/ownership tests; supplied opportunities are NOT native legality proof."""
import numpy as np
import pytest
from lupa.lua51 import LuaError

from soku_rl.policy.god.query import QueryTeacher
from soku_rl.policy.god.observed_memory import classify
from test_teacher_observed_memory import actor as passive_actor, observation
from test_teacher_perception import row
from test_teacher_intents import boundary


def actor(script='01_marisa_main.ai',context=False):
    base=passive_actor(script,instruction_limit=1000000).teacher
    settings=base.spec | {'decisions':'source_scoped_v1'}
    if context:settings['task_context']=dict(fixed_weather='clear',cards_disabled=True)
    return QueryTeacher(base.interface,settings).spawn(17)


def state(script='01_marisa_main.ai',action=0,enemy_action=0):
    current=row()
    current[4]=1 if script.startswith('01_') else 0
    current[14]=1-current[4]
    current[5]=action;current[15]=enemy_action
    current[9]=current[19]=0
    return current


def warm(episode,current=None):
    current=state(episode.teacher.spec['script']) if current is None else current
    for _ in range(episode.teacher.spec['reaction_frames']+1):
        answer=episode.query(observation(current))
    assert answer.status=='WAIT_TRUSTED_DECISION_BOUNDARY'
    return current


def decide(episode,current):
    return episode.query(observation(current),opportunity=boundary(),boundary_id=episode.frame+1)


def fixture(episode,code):
    """Inject a labelled test definition, never claim it is original God AI."""
    scoped=episode.scoped_decisions
    scoped.prepare()
    scoped.loading(True)
    try:
        scoped.run(scoped.compile(code.encode('ascii'),b'@fixture/scoped',scoped.env))
    finally:
        scoped.loading(False)


@pytest.mark.parametrize('script',['01_marisa_main.ai','00_reimu_main.ai'])
def test_original_decision_missing_state_is_local_and_source_memory_survives(script):
    episode=actor(script);current=warm(episode)
    first=decide(episode,current)
    assert first.status=='MASKED_SOURCE_STATE' and first.action is None
    assert 'UNTRUSTED_TEACHER_STATE:' in first.detail
    assert episode.blocked is None and episode.scoped_decisions.fatal is None
    vm=episode.lua
    before=episode.observed_memory.observations
    for _ in range(4):
        result=decide(episode,current)
        assert result.status=='MASKED_SOURCE_STATE' and result.action is None
    assert episode.lua is vm and episode.observed_memory.observations==before+4
    assert not episode.intents.loaded  # legacy initializer/global rollback never used
    assert episode.scoped_decisions.loaded and episode.scoped_decisions.queries==5
    assert episode.api.requested==[0]*10  # no original key callback operates the game


@pytest.mark.parametrize('script',['01_marisa_main.ai','00_reimu_main.ai'])
def test_original_source_changes_mask_reason_with_real_delayed_state_not_cached_error(script):
    episode=actor(script,context=True);current=warm(episode)
    ordinary=decide(episode,current)
    assert ordinary.status=='MASKED_SOURCE_STATE'
    original_missing=ordinary.detail
    wake=state(script,action=97)
    for _ in range(episode.teacher.spec['reaction_frames']+1):
        answer=decide(episode,wake)
    assert answer.status=='UNSUPPORTED_INTENT_SEGMENT'
    for _ in range(episode.teacher.spec['reaction_frames']+1):
        answer=decide(episode,current)
    assert answer.status=='MASKED_SOURCE_STATE'
    assert answer.detail==original_missing
    assert episode.blocked is None and episode.scoped_decisions.fatal is None


def test_source_static_parameters_are_not_native_frame_data_or_mutable_memory():
    episode=actor();current=warm(episode)
    decide(episode,current)
    scoped=episode.scoped_decisions
    _,static=scoped.state()
    assert static[b'stand_neutral']==34 and static[b'caution_range']==320
    assert static[b'rapid'][300]==8 and static[b'rapid'][410]==25
    assert scoped.report()['definition_regions']['lines']==[[163,403],[418,639]]
    assert scoped.quarantined>=1
    with pytest.raises(LuaError,match='READ_ONLY_TEACHER_MEMORY'):
        episode.lua.eval(b'function(t) t[300]=999 end')(static[b'rapid'])
    # Unused source flags are constant0, not a claim about student held keys.
    assert scoped.env[b'RnI_B']==0 and scoped.env[b'RnI_C']==0
    assert episode.lua.eval(b'function(e) return rawget(e,"RnI_B") end')(episode.env) is None
    assert classify('rapid')[0]=='STATIC_AI_PARAMETER'
    assert classify('true_math')[0]=='QUERY_SCRATCH'


def test_task_weather_is_explicit_constraint_not_native_or_hand_read():
    episode=actor(context=True);current=warm(episode)
    decide(episode,current)
    scoped=episode.scoped_decisions
    assert scoped.env[b'weather']==21 and scoped.env[b'weather2']==21
    assert scoped.env[b'weather_time']==0
    assert episode.lua.eval(b'function(e)return rawget(e,"weather"),rawget(e,"my_card")end')(episode.env)==(None,None)
    assert scoped.report()['task_premises']['native_verified'] is False
    ordinary=actor();warm(ordinary);ordinary.scoped_decisions.prepare()
    with pytest.raises(LuaError,match='UNTRUSTED_TEACHER_STATE:weather'):
        ordinary.scoped_decisions.env[b'weather']
    assert ordinary.scoped_decisions.task_gates==[]
    assert len(scoped.task_gates)>2
    assert any(g['line']==497 for g in scoped.task_gates)
    with pytest.raises(LuaError,match='UNTRUSTED_TEACHER_STATE:get_card_id'):
        scoped.env[b'get_card_id']


def test_pure_guard_reorder_preserves_boolean_for_complete_numeric_inputs():
    episode=actor();warm(episode);episode.scoped_decisions.prepare()
    seam=episode.scoped_decisions.guard_adaptation
    old=seam['original'].removesuffix(' and')
    new=seam['adapted'].removesuffix(' and')
    functions=[episode.lua.eval(('function(y,my_yspeed,my_act)return '+text+' end').encode()) for text in (old,new)]
    for action in (-1,0,49,50,97,107,108,400):
        for y in (0,1,30,200):
            for speed in (-20,0,10):
                assert functions[0](y,speed,action)==functions[1](y,speed,action)
    for seam in episode.scoped_decisions.flag_guard_adaptations:
        functions=[episode.lua.eval(('function(my_act,y,my_airEX,flags) local function get_fflags()return flags end;return '+seam[k]+' end').encode()) for k in ('original','adapted')]
        for action in (0,299,300,400):
            for y in (0,100):
                for air in (0,1,2):
                    for flags in (False,True):
                        assert functions[0](action,y,air,flags)==functions[1](action,y,air,flags)


def test_original_observed_reimu_flags_recover_without_execution_assumptions():
    episode=actor('00_reimu_main.ai');warm(episode)
    scoped=episode.scoped_decisions
    assert scoped.tactical=={'ex_combo':0,'crush_combo':0}
    # No teacher plan is used to set flag1. Missing limit/weather prevents the
    # corresponding native-dependent source guard from being treated as false.
    episode.env[b'my_act']=0;episode.env[b'enemy_act']=55
    scoped.observe(scoped.last_observation+1)
    assert not scoped.tactical
    episode.env[b'my_act']=601
    scoped.observe(scoped.last_observation+1)
    assert scoped.tactical=={'ex_combo':2}  # source61..63 observes actual action
    episode.env[b'my_act']=401
    scoped.observe(scoped.last_observation+1)
    assert scoped.tactical=={'ex_combo':0}
    scoped.invalidate_observation()
    assert not scoped.tactical


def test_rejected_fixture_intent_discards_plan_not_observed_or_static_state():
    episode=actor('00_reimu_main.ai');current=warm(episode)
    fixture(episode,'''function teacher_decide()
      ex_combo=1;true_math=9;command('236B');error('STALE_COROUTINE_RESUMED')
    end''')
    vm=episode.lua
    first=decide(episode,current)
    assert first.status=='INTENT_PROPOSED' and first.action is None
    assert episode.scoped_decisions.last_plan=={'ex_combo':1}
    assert episode.scoped_decisions.tactical=={'ex_combo':0,'crush_combo':0}
    _,static=episode.scoped_decisions.state()
    rapid=static[b'rapid']
    episode.intents.reject(first.intent_id)
    assert not episode.scoped_decisions.last_plan
    second=decide(episode,current)
    assert second.intent=='236B' and second.intent_id==2 and episode.lua is vm
    assert episode.scoped_decisions.tactical=={'ex_combo':0,'crush_combo':0}
    assert episode.lua.eval(b'function(a,b)return a==b end')(rapid,episode.scoped_decisions.state()[1][b'rapid'])
    assert episode.blocked is None


@pytest.mark.parametrize('operation',[
    'my_preact[0]=999', 'math.randomseed=999', 'unknown_future_flag=1',
    'local x=enemy_obj_n', 'local x=enemy_yspeed'])
def test_untrusted_write_or_read_masks_current_segment_then_can_replan(operation):
    episode=actor();current=warm(episode)
    fixture(episode,'function teacher_decide() '+operation+";command('5A') end")
    first=decide(episode,current)
    assert first.status=='MASKED_SOURCE_STATE' and episode.blocked is None
    assert not episode.scoped_decisions.last_plan
    assert episode.observed_memory.histories['my_preact'][0]==0
    fixture(episode,"function teacher_decide() command('5A') end")
    next=decide(episode,current)
    assert next.status=='INTENT_PROPOSED' and next.intent=='5A'


def test_input_operation_yields_local_mask_without_executing_continuation():
    episode=actor();current=warm(episode)
    fixture(episode,"function teacher_decide() ex_combo=1;wait(30);error('STALE') end")
    result=decide(episode,current)
    assert result.status=='UNSUPPORTED_INTENT_SEGMENT' and result.detail=='wait'
    assert not episode.scoped_decisions.last_plan and episode.blocked is None
    fixture(episode,"function teacher_decide() lvr236(ACT_B) end")
    result=decide(episode,current)
    assert result.intent=='236B' and result.action is None


def test_instances_and_reset_do_not_share_source_definitions_rng_or_plans():
    first,second=actor(),actor()
    current=warm(first);warm(second)
    fixture(first,"function teacher_decide() ex_combo=1;command('236B') end")
    decide(first,current)
    assert first.scoped_decisions.last_plan=={'ex_combo':1}
    assert not second.scoped_decisions.loaded and not second.scoped_decisions.last_plan
    reset=first.teacher.spawn(17)
    assert reset.lua is not first.lua and not reset.scoped_decisions.loaded
    assert not reset.scoped_decisions.tactical


def test_unexpected_source_failure_requires_reset_not_fake_recovery():
    episode=actor();current=warm(episode)
    fixture(episode,"function teacher_decide() error('UNEXPECTED_SOURCE_FAILURE') end")
    result=decide(episode,current)
    assert result.status=='BLOCKED_SCOPED_SOURCE'
    fixture(episode,"function teacher_decide() command('5A') end")
    assert decide(episode,current).status=='BLOCKED_SCOPED_SOURCE'


def test_candidates_or_forced_state_do_not_enable_scoped_decision():
    episode=actor();current=warm(episode)
    for opportunity in (boundary(source='MASKED_CANDIDATES_NATIVE_AUTHORITY'),boundary(forced=True)):
        result=episode.query(observation(current),opportunity=opportunity,boundary_id=episode.frame+1)
        assert result.status=='WAIT_TRUSTED_DECISION_BOUNDARY'
    assert not episode.scoped_decisions.loaded and episode.scoped_decisions.queries==0


def test_scoped_profile_preserves1832_and_training_attachment_guard(monkeypatch):
    import sys
    from pathlib import Path
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from soku_rl.env.hisouten_env import EpisodeConfig
    from soku_rl.env.wrappers.learning import LearningConfig,LearningInterface
    from soku_rl.rl.kickstarting import attach_kickstarting
    root=Path(__file__).resolve().parents[2]
    for key,value in dict(SOKURL_RL_ROOT=root/'rl',SOKURL_PROJECT_ROOT=root,
        SOKURL_PLATFORM_ROOT=root/'platform',SOKURL_GAME_DIR=root/'th123_jp',
        SOKURL_PYTHON=sys.executable,SOKURL_WORLD_STUDENT=root/'rl/runs/teacher-world-20261018/student').items():
        monkeypatch.setenv(key,str(value))
    with initialize_config_dir(config_dir=str(root/'rl/config'),version_base='1.3'):
        config=OmegaConf.to_container(compose(config_name='train_combat_kickstarting_scoped'),resolve=True)
    interface=LearningInterface(EpisodeConfig.from_dict(config['episode']),LearningConfig(**config['wrappers']))
    assert interface.observation_space.shape==(1832,)
    assert interface.action_space.n==90
    assert config['rl']['kickstarting']['teacher']['decisions']=='source_scoped_v1'
    with pytest.raises(RuntimeError,match='trusted decision-boundary provider'):
        attach_kickstarting(None,None,config['rl']['kickstarting'],False)


def test_source_constant_assumptions_require_reviewed_api_identity():
    episode=actor()
    episode.teacher.package.api+=b'\n-- different API version'
    with pytest.raises(ValueError,match='reviewed original API source'):
        episode.teacher.spawn(18)


def test_corrupted_gate_binding_fails_without_fallback_or_source_execution():
    import copy
    episode=actor(context=True);current=warm(episode)
    episode.scoped_decisions.registry=copy.deepcopy(episode.scoped_decisions.registry)
    gate=next(g for g in episode.scoped_decisions.registry['gates'] if g['source']=='01_marisa_main.ai')
    gate['condition_sha256']='0'*64
    answer=decide(episode,current)
    assert answer.status=='BLOCKED_SCOPED_SOURCE'
    assert 'Invalid card-only source condition binding' in answer.detail
    assert episode.api.requested==[0]*10

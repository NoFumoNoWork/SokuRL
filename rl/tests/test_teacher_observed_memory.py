"""Passive original-AI memory must follow reality, never a rejected Lua plan."""
from dataclasses import replace

import numpy as np
import pytest
from lupa.lua51 import LuaError

from soku_rl.env.wrappers.learning import LearningInterface
from soku_rl.policy.god.observed_memory import RULES, classify
from soku_rl.policy.god.query import Advice, QueryTeacher
from soku_rl.policy.god.source_adapter import SOURCES, digest
from test_kickstarting import spec
from test_teacher_perception import contract, row
from test_teacher_source_adapter import PACKAGE, package


def actor(script='01_marisa_main.ai',instruction_limit=100000):
    package()
    base=contract()
    interface=LearningInterface(base.episode,replace(base.config,action_history=8))
    policy=QueryTeacher(interface,spec(PACKAGE) | dict(script=script,mode='intent',
        source_adapter='th123ai_20110925',perception='student_visible_v1',
        memory='observed_source_v1',instruction_limit=instruction_limit))
    return policy.spawn(17)


def observation(current):
    return np.r_[np.tile(current,8),np.zeros(64,np.float32)]


def ready(episode,current=None):
    current=row() if current is None else current
    for _ in range(episode.teacher.spec['reaction_frames']+1):
        result=episode.query(observation(current))
    assert result.status=='PHASE2_OBSERVED_MEMORY_READY' and result.action is None
    return episode.observed_memory


def observe(memory,index,**fields):
    for name,value in fields.items():memory.episode.env[name.encode()]=value
    return memory.observe(index)


@pytest.mark.parametrize('script',['01_marisa_main.ai','00_reimu_main.ai'])
def test_exact_source_rules_with_no_original_main_restart_or_global_rollback(script):
    episode=actor(script)
    memory=ready(episode)
    assert not episode.intents.loaded and not episode.intents.initialized
    assert episode.blocked is None and memory.observations==1
    original=episode.teacher.package.files['def_clc.lua']
    assert digest(original)==SOURCES['def_clc.lua']
    for rule in RULES:
        assert memory.slice(rule.first,rule.last).splitlines()[rule.first-1:]==original.splitlines()[rule.first-1:rule.last]
    assert memory.values['my_frontside']==940 and memory.values['my_backside']==260
    assert memory.values['air_math']==0
    # Joining a snapshot does not prove previous actions, ground duration, or
    # original AI initialization180. None are filled with fabricated zeros.
    for name in ('graund_math','my_allframe','enemy_allframe','after_guard_count'):
        assert name not in memory.values
    assert memory.report()['histories']['my_preact']=={'0':400}
    assert episode.lua.eval(b'function(e) return rawget(e,"RnI_B"),rawget(e,"bio_math"),rawget(e,"enemy_obj_n") end')(episode.env)==(None,None,None)


def test_history_uses_actual_action_and_frame_rewinds_in_original_order():
    episode=actor();memory=ready(episode)
    observe(memory,1,my_act=401,my_frame=0)
    assert memory.values['my_allframe']==1
    assert memory.histories['my_preact'][1]==400
    observe(memory,2,my_frame=1)
    assert memory.values['my_allframe']==2
    observe(memory,3,my_frame=0)  # same action, real observed restart
    assert memory.values['my_allframe']==3
    assert memory.histories['my_preframe'][1]==1
    # A proposed but unexecuted attack does not change the actual history.
    episode.intents.pending=Advice('INTENT_PROPOSED',intent='236B',intent_id=1)
    vm,table=episode.lua,memory.histories['my_preact']
    episode.intents.reject(1)
    observe(memory,4,my_frame=1)
    assert episode.lua is vm and memory.histories['my_preact'] is table
    assert memory.values['my_allframe']==4 and memory.histories['my_preact'][0]==401
    assert episode.intents.invalidations==1 and episode.blocked is None


def test_unknown_groups_recover_without_resetting_independent_memories():
    from soku_rl.env.observation.animation import frame_width
    airborne=row();airborne[frame_width(16)+3]=40
    memory=ready(actor(),airborne)
    assert 'air_math' not in memory.values
    observe(memory,1,ey=40,my_act=150,my_frame=0)
    assert memory.values['graund_math']==0
    assert memory.values['after_guard_count']==60
    assert 'air_math' not in memory.values
    observe(memory,2,ey=0,my_act=0,my_frame=3)
    assert memory.values['air_math']==0 and memory.values['graund_math']==1
    assert memory.values['after_guard_count']==59


def test_passive_counters_are_not_native_timers_and_recover_after_gap():
    memory=ready(actor())
    observe(memory,1,ey=40,my_act=150,my_frame=0)
    assert memory.values['air_math']==1 and memory.values['after_guard_count']==60
    observe(memory,2,ey=40,my_frame=1)
    assert memory.values['air_math']==2 and memory.values['after_guard_count']==60
    observe(memory,3,ey=0,my_act=0,my_frame=2)
    assert memory.values['graund_math']==1 and memory.values['after_guard_count']==59
    report=observe(memory,5,ey=40,my_act=0,my_frame=3)
    assert report['gaps']==1 and 'air_math' not in memory.values
    assert memory.values['graund_math']==0 and 'after_guard_count' not in memory.values
    assert memory.histories['my_preact'][1] is None
    observe(memory,6,ey=0,my_act=150,my_frame=0)
    assert memory.values['air_math']==0 and memory.values['after_guard_count']==60
    observe(memory,7,ey=10,my_act=0,my_frame=3)
    assert memory.values['air_math']==1 and memory.values['after_guard_count']==59
    assert report['semantics']['clock'].endswith('not native battle_time')


def test_readonly_memory_tables_do_not_expose_unknown_slots_as_false():
    episode=actor();memory=ready(episode)
    with pytest.raises(LuaError,match='UNTRUSTED_TEACHER_MEMORY:1'):
        episode.lua.eval(b'function(e) return e.my_preact[1] end')(episode.env)
    with pytest.raises(LuaError,match='READ_ONLY_TEACHER_MEMORY'):
        episode.lua.eval(b'function(e) e.my_preact[0]=999 end')(episode.env)
    assert memory.histories['my_preact'][0]==400


def test_local_invalidation_preserves_vm_and_isolates_reset_and_other_worker():
    first,second=actor(),actor()
    memory=ready(first)
    vm=first.lua
    first.intents.pending=Advice('INTENT_PROPOSED',intent='236B',intent_id=1)
    memory.interrupt('TEST_VISIBILITY_GAP')
    assert first.lua is vm and first.intents.pending is None
    assert first.intents.invalidations==1 and not memory.histories
    assert first.lua.eval(b'function(e) return rawget(e,"my_preact") end')(first.env) is None
    assert not second.observed_memory.histories
    observe(memory,3,my_act=0,my_frame=0,ey=0)
    assert memory.last_index==3 and first.blocked is None
    reset=first.teacher.spawn(17)
    assert reset.lua is not vm and not reset.observed_memory.histories


def test_delayed_fields_not_future_observation_and_visibility_gap():
    episode=actor();initial=row();initial[5]=0;initial[9]=0
    ready(episode,initial)
    new=row();new[5]=401;new[9]=0
    episode.query(observation(new))
    assert episode.observed_memory.histories['my_preact'][0]==0
    episode.query(observation(new));episode.query(observation(new))
    assert episode.observed_memory.histories['my_preact'][0]==401
    missing=new.copy();missing[0]=0
    for _ in range(3):answer=episode.query(observation(missing))
    assert answer.status=='UNAVAILABLE_VISIBLE_FIGHTER'
    assert episode.observed_memory.last_index is None
    for _ in range(3):answer=episode.query(observation(new))
    assert answer.status=='PHASE2_OBSERVED_MEMORY_READY' and episode.blocked is None


def test_budget_failure_and_rewind_not_masked_as_missing_memory():
    episode=actor();memory=ready(episode)
    with pytest.raises(ValueError,match='repeat or rewind'):memory.observe(0)
    memory.runners['air_ticks']=memory.compile(b'while true do end',b'@negative/infinite')
    with pytest.raises(LuaError,match='QUERY_INSTRUCTION_LIMIT'):memory.observe(1)
    with pytest.raises(RuntimeError,match='BLOCKED_PASSIVE_MEMORY_FAILURE'):memory.observe(2)


def test_memory_roles_do_not_imply_trust_or_native_equivalence():
    assert classify('after_guard_count')[0]=='EVENT_MEMORY'
    assert classify('RnI_B')[0]=='EXECUTION_PROGRESS'
    assert classify('rapid')[0]=='TACTICAL_INTENT'
    assert classify('enemy_obj_id')[1]=='UNTRUSTED_MISSING_SHARED_DEPENDENCIES'
    assert classify('new_unknown')[0]=='UNKNOWN'


def test_missing_read_drops_only_its_outputs_without_partial_commit():
    episode=actor();memory=ready(episode)
    memory.runners['air_ticks']=memory.compile(b'air_math=999; air_math=unavailable_hidden_value',b'@negative/partial')
    observe(memory,1,my_act=150,my_frame=0)
    assert 'air_math' not in memory.values and episode.blocked is None
    assert memory.values['after_guard_count']==60
    assert memory.status['air_ticks']['status']=='UNTRUSTED_DEPENDENCY'
    assert episode.lua.eval(b'function(e) return rawget(e,"air_math") end')(episode.env) is None


def test_unreviewed_memory_source_rejected_without_changing_community_files():
    from soku_rl.policy.god.observed_memory import ObservedMemory
    episode=actor()
    files=dict(episode.teacher.package.files)
    files['def_clc.lua']+=b'\n-- different source'
    from types import SimpleNamespace
    episode.teacher.package=SimpleNamespace(files=files)
    with pytest.raises(ValueError,match='Unreviewed passive-memory source'):ObservedMemory(episode)


def test_guard_audit_definition_loading_cannot_overwrite_observed_memory():
    from audit_god_teacher import inspect_guards
    # Original definition loading initializes large object arrays. Use the
    # explicit production-profile audit budget; passive-memory tests use100k.
    episode=actor(instruction_limit=1000000);memory=ready(episode)
    observe(memory,1,my_act=401,my_frame=0)
    before=memory.report()
    current=episode.env[b'my_preact']
    registry=dict(sources=SOURCES,functions={},branches=[])
    result=inspect_guards(episode,registry)
    assert result['isolation'].startswith('definitions evaluated in a disposable audit VM')
    assert memory.report()==before and episode.env[b'my_preact'][0]==401
    assert episode.lua.eval(b'function(a,b) return a==b end')(current,episode.env[b'my_preact'])
    assert not episode.intents.loaded


def test_memory_profile_and_training_gate_stay_explicit(monkeypatch):
    import sys
    from pathlib import Path
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    root=Path(__file__).resolve().parents[2]
    for key,value in dict(SOKURL_RL_ROOT=root/'rl',SOKURL_PROJECT_ROOT=root,
        SOKURL_PLATFORM_ROOT=root/'platform',SOKURL_GAME_DIR=root/'th123_jp',
        SOKURL_PYTHON=sys.executable,SOKURL_WORLD_STUDENT=root/'rl/runs/teacher-world-20261018/student').items():
        monkeypatch.setenv(key,str(value))
    with initialize_config_dir(config_dir=str(root/'rl/config'),version_base='1.3'):
        config=OmegaConf.to_container(compose(config_name='train_combat_kickstarting_memory'),resolve=True)
    settings=config['rl']['kickstarting']['teacher']
    assert settings['memory']=='observed_source_v1'
    from soku_rl.rl.kickstarting import attach_kickstarting
    with pytest.raises(RuntimeError,match='trusted decision-boundary provider'):
        attach_kickstarting(None,None,config['rl']['kickstarting'],False)

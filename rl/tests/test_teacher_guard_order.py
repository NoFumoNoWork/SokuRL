"""Pure guard rewrites must not grant missing skills or cross effect barriers."""
import copy
import json
from pathlib import Path
import subprocess
import sys

from lupa.lua51 import LuaError, LuaRuntime
import pytest

from soku_rl.policy.god.guard_order import load_registry, replacements
from test_teacher_scoped_state import actor, decide, warm


ROOT = Path(__file__).resolve().parents[2]


def test_original_guard_registry_exact_reproduction(tmp_path):
    site = ROOT/'rl/.tools/lua-parser/site'
    if not (site/'luaparser/ast.py').is_file(): pytest.skip('optional isolated AST dependency')
    output = tmp_path/'regenerated.json'
    subprocess.run([sys.executable, '-B', '-I', '-S', str(ROOT/'rl/tools/teacher_guard_order.py'),
        '--site', str(site), '--output', str(output)], check=True, timeout=90)
    registry, _ = load_registry()
    assert json.loads(output.read_text('utf-8')) == registry
    assert registry['guards']
    assert {(r['source'],r['line']) for r in registry['guards']} >= {
        ('01_marisa_main.ai',518),('00_reimu_main.ai',583)}


def test_complete_numeric_guard_truth_rng_and_line_count_equivalence():
    registry, _ = load_registry()
    lua = LuaRuntime(encoding=None)
    compile_guard = lua.eval(b'''function(code)
      local set= setfenv
      local fn=assert(loadstring('return function() return not not ('..code..') end'))()
      return function(env) set(fn,env);return fn() end
    end''')
    check = lua.eval(b'''function(original,adapted,action,height,skill,enemy,enemy_y,weather)
      local function evaluate(fn)
        local calls=0
        local env=setmetatable({my_act=action,enemy_act=enemy,y=height,ey=enemy_y,weather=weather,
          enemy_yspeed=-1,e_gs=0.5,
          dis=90,is_dir_front=true,enemy_allframe=2,get_skill_lv=function()return skill end,
          math={random=function()calls=calls+1;return 1 end}},
          {__index=function() return 0 end}) -- complete numeric TEST FIXTURE only
        return fn(env),calls
      end
      local a,ar=evaluate(original);local b,br=evaluate(adapted)
      return a==b and ar==br
    end''')
    for source in ('01_marisa_main.ai','00_reimu_main.ai'):
        data=(ROOT/'third_party/th123_ai/package/th123ai/script'/source).read_bytes()
        for first,last,code in replacements(data,source,registry):
            assert code.count(b'\n') == data[first:last].count(b'\n')
            old,new=map(compile_guard,(data[first:last],code))
            for action in (0,5,6,50,300,400):
                for height in (0,100):
                    for skill in (-1,0,1,4):
                        for enemy in (0,300):
                            for enemy_y in (0,100):
                                for weather in (9,21):
                                    assert check(old,new,action,height,skill,enemy,enemy_y,weather)


@pytest.mark.parametrize('script',['01_marisa_main.ai','00_reimu_main.ai'])
def test_missing_skill_not_zero_filled_and_irrelevant_branch_can_be_disproved(script):
    episode=actor(script,context=True);current=warm(episode)
    answer=decide(episode,current)
    assert answer.status in ('MASKED_SOURCE_STATE','UNSUPPORTED_INTENT_SEGMENT','INTENT_PROPOSED')
    assert 'get_skill_lv' not in answer.detail
    with pytest.raises(LuaError,match='UNTRUSTED_TEACHER_STATE:get_skill_lv'):
        episode.scoped_decisions.env[b'get_skill_lv']
    assert episode.api.requested==[0]*10
    assert episode.scoped_decisions.report()['pure_guard_order']['guards']


def test_relevant_skill_branch_still_masks_before_rng_or_any_key_submission():
    registry,_=load_registry()
    source='01_marisa_main.ai'
    data=(ROOT/'third_party/th123_ai/package/th123ai/script'/source).read_bytes()
    row=next(r for r in registry['guards'] if r['source']==source and r['line']==518)
    code=next(code for start,_,code in replacements(data,source,registry) if start==row['start'])
    lua=LuaRuntime(encoding=None)
    fn=lua.eval(b'''function(code)
      local f=assert(loadstring(code));local calls=0
      local env=setmetatable({dis=50,y=0,my_act=0,enemy_act=300,enemy_allframe=0,
        weather=21,is_dir_front=true,math={random=function() calls=calls+1;return 1 end}},
        {__index=function(_,k) error('MISSING:'..k) end})
      setfenv(f,env);local ok,err=pcall(f);return ok,err,calls
    end''')
    ok,error,calls=fn(b'return '+code)
    assert not ok and b'MISSING:get_skill_lv' in error and calls==0


def test_corrupted_order_registry_is_fatal_not_fallback():
    episode=actor(context=True);current=warm(episode)
    registry=copy.deepcopy(episode.scoped_decisions.pure_guards)
    row=next(r for r in registry['guards'] if r['source']=='01_marisa_main.ai')
    row['sha256']='0'*64
    episode.scoped_decisions.pure_guards=registry
    answer=decide(episode,current)
    assert answer.status=='BLOCKED_SCOPED_SOURCE'
    assert 'Invalid pure-guard' in answer.detail
    assert episode.api.requested==[0]*10


def test_generator_stops_at_rng_unknown_calls_indexing_and_shadowed_locals(tmp_path):
    site=ROOT/'rl/.tools/lua-parser/site'
    if not (site/'luaparser/ast.py').is_file(): pytest.skip('optional isolated AST dependency')
    package=tmp_path/'package';package.mkdir()
    text='''function main()
 if get_skill_lv(0,1)>=1 and y==0 and my_act<=5 then end -- pure
 if get_skill_lv(0,1)>=1 and math.random(16)==1 and y==0 then end -- rng
 if get_skill_lv(0,1)>=1 and unknown() and y==0 then end -- unknown
 if get_skill_lv(0,1)>=1 and 1/0>0 and y==0 then end -- division
 if get_skill_lv(0,1)>=1 and my_preact[0]==0 and y==0 then end -- indexing
 if get_skill_lv(0,1)>=1 and y==0 or enemy_act==0 then end -- or_root
 if get_skill_lv(0,1)>=1 and y==0 and unknown() and x>0 then end -- partial
end
'''
    (package/'01_marisa_main.ai').write_text(text,encoding='ascii')
    shadow=text.replace('function main()', 'function main() local y=0')
    (package/'00_reimu_main.ai').write_text(shadow,encoding='ascii')
    for name in ('def_clc.lua','function_command.txt'):
        (package/name).write_text('-- test only',encoding='ascii')
    api=tmp_path/'api.ai';api.write_text('-- test only',encoding='ascii')
    # Isolate the optional parser even for negative controls. Fixture source
    # pins apply only to this child, never the production runtime or package.
    script='''import hashlib,importlib.util,json,sys,types
from pathlib import Path
sys.path.insert(0,sys.argv[1])
p=types.ModuleType('fixture_guard');p.__path__=[sys.argv[2]];sys.modules[p.__name__]=p
s=importlib.util.spec_from_file_location('fixture_guard.guard_order',Path(sys.argv[2])/'guard_order.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('generator',sys.argv[3]);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
package=Path(sys.argv[4]);m.SOURCES={k:hashlib.sha256((package/k).read_bytes()).hexdigest() for k in m.SOURCES}
result=g.generate(package,Path(sys.argv[5]),m)
Path(sys.argv[6]).write_text(json.dumps(result),encoding='utf-8')
'''
    output=tmp_path/'negative.json'
    subprocess.run([sys.executable,'-B','-I','-S','-c',script,str(site),
        str(ROOT/'rl/src/soku_rl/policy/god'),str(ROOT/'rl/tools/teacher_guard_order.py'),
        str(package),str(api),str(output)],check=True,timeout=60)
    guards=json.loads(output.read_text('utf-8'))['guards']
    assert [r['line'] for r in guards]==[2,8]
    assert all(r['source']=='01_marisa_main.ai' for r in guards)
    assert guards[0]['order']==[1,2,0]
    assert guards[1]['order']==[1,0,2,3]  # unknown call remains a barrier


def test_permutation_cannot_be_changed_to_cross_a_proven_effect_barrier():
    registry,_=load_registry()
    source='01_marisa_main.ai'
    data=(ROOT/'third_party/th123_ai/package/th123ai/script'/source).read_bytes()
    registry=copy.deepcopy(registry)
    row=next(r for r in registry['guards'] if r['source']==source)
    barrier=next(i for i,c in enumerate(row['clauses']) if c['kind']=='BARRIER')
    row['order'][0],row['order'][barrier]=row['order'][barrier],row['order'][0]
    with pytest.raises(ValueError,match='effect barrier/order'):
        replacements(data,source,registry)


def test_special_api_only_returns_character_inapplicability_not_hidden_values():
    from sokurl_platform.native_layout import SPECIAL_FIELDS
    episode=actor(context=True);warm(episode);episode.scoped_decisions.prepare()
    scoped=episode.scoped_decisions
    get=scoped.env[b'get_special_data']
    # All applicability rows reuse the existing Platform mapping, not a list
    # of action IDs or guessed timers. They never access a native observation.
    for index,(required,_,_) in SPECIAL_FIELDS.items():
        if required<0: continue
        for side,name in ((0,b'my_char'),(1,b'enemy_char')):
            episode.env[name]=1 if required!=1 else 0
            assert get(side,index)==-1
            episode.env[name]=required
            with pytest.raises(ValueError,match='applicable_or_unreviewed_field'):
                get(side,index)
    for index in (4,7,9,13,14,99):
        with pytest.raises(ValueError,match='applicable_or_unreviewed_field'):
            get(0,index)
    assert scoped.report()['special_applicability']['native_values_exposed'] is False
    assert episode.api.requested==[0]*10

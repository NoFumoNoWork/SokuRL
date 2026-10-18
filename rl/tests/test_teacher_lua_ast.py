"""AST dependencies must include helper calls and previous false guards."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT=Path(__file__).resolve().parents[2]
SITE=ROOT/'rl/.tools/lua-parser/site'


def test_isolated_ast_scope_priority_comments_and_source_offsets(tmp_path):
    if not (SITE/'luaparser/ast.py').is_file():pytest.skip('optional isolated AST auditor not prepared')
    package=tmp_path/'package';package.mkdir()
    text='''-- enemy_xspeed in this comment is NOT a dependency
function helper(p)
 if p==1 and enemy_obj_n==0 then return true end
end
function main()
 print("\u80fdx")
 if my_act==0 then command('5A')
 elseif helper(1) then command('236B')
 else command('5C') end
end
function loop_check()
 for i=1,enemy_obj_n do
  if my_act==0 then command('5A') end
 end
end
'''
    (package/'01_marisa_main.ai').write_bytes(text.encode('cp932'))
    for name in ('00_reimu_main.ai','def_clc.lua','function_command.txt'):
        (package/name).write_text('-- empty fixture',encoding='ascii')
    api=tmp_path/'api.ai';api.write_text('ACT_A=4',encoding='ascii')
    allowed=tmp_path/'available.json';allowed.write_text(json.dumps(['my_act']),encoding='utf-8')
    output=tmp_path/'branches.json'
    subprocess.run([sys.executable,'-B','-I','-S',str(ROOT/'rl/tools/teacher_lua_ast.py'),
        '--site',str(SITE),'--package',str(package),'--api',str(api),
        '--available',str(allowed),'--output',str(output)],check=True,timeout=120)
    result=json.loads(output.read_text('utf-8'))
    first=next(x for x in result['branches'] if x['line']==7)
    assert first['guard_status']=='DEPENDENCY_READY_ONLY' and first['path_status']=='DEPENDENCY_READY_ONLY'
    raw=(package/'01_marisa_main.ai').read_bytes()
    assert raw[first['condition_start']:first['condition_end']].decode('cp932')==first['condition']
    second=next(x for x in result['branches'] if x['line']==8)
    assert 'enemy_obj_n' in second['guard_reads'] and 'enemy_xspeed' not in second['guard_reads']
    assert 'p' not in second['guard_reads']
    otherwise=next(x for x in result['branches'] if x['kind']=='Else')
    assert otherwise['path_status']=='UNSUPPORTED_PATH'
    assert any(x['name']=='enemy_obj_n' for x in otherwise['missing_path'])
    assert all(x['execution_status'].startswith('NOT_ENABLED') for x in result['branches'])
    in_loop=next(x for x in result['branches'] if x['enclosing_function']=='loop_check')
    assert in_loop['path_status']=='UNSUPPORTED_PATH'
    assert in_loop['missing_loops'][0]['name']=='enemy_obj_n'
    # Child -S parser must not replace or import into the training interpreter.
    from importlib.metadata import version
    assert version('antlr4-python3-runtime').startswith('4.9.')
    assert 'luaparser' not in sys.modules
    from test_kickstarting import interface,spec,observation,API
    from soku_rl.policy.god.query import QueryTeacher
    if not API.is_file():pytest.skip('external original API not prepared')
    from audit_god_teacher import inspect_guards
    contract=interface()
    actor=QueryTeacher(contract,spec(package) | {'mode':'intent','script':'01_marisa_main.ai'}).spawn(17)
    for _ in range(3):actor.query(observation(contract,1))
    probe=inspect_guards(actor,result)
    last=next(x for x in probe['branches'] if x['id'].endswith(':else'))
    assert last['path']=='UNKNOWN_PRECEDING_OR_ENCLOSING_GUARD'
    assert all(x['atomic_label'] is None for x in probe['branches'])

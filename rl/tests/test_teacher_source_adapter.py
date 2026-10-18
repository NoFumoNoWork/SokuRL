"""Static/Lua-only original source adaptation; no native gameplay observations."""
import hashlib
from types import SimpleNamespace

import pytest
from lupa.lua51 import LuaRuntime, LuaError

from soku_rl.policy.god.query import QueryTeacher
from soku_rl.policy.god.source_adapter import CommunityIntentAdapter, SOURCES, TRANSACTION
from soku_rl.policy.god.api import ScriptAPI
from soku_rl.policy.god.package import ScriptPackage
from test_kickstarting import ROOT, API, interface, observation, spec
from test_teacher_intents import actor, boundary, warm


PACKAGE = ROOT / 'third_party/th123_ai/package/th123ai/script'


def package():
    if not API.is_file() or not PACKAGE.is_dir():
        pytest.skip('external community package unavailable')
    return ScriptPackage(PACKAGE, API)


@pytest.mark.parametrize('script', ['01_marisa_main.ai', '00_reimu_main.ai'])
def test_pinned_split_compiles_and_preserves_original_line_bodies(script):
    original = package()
    adapter = CommunityIntentAdapter(original, script)
    lua = LuaRuntime(encoding=None, unpack_returned_tuples=True)
    parse = lua.eval(b"function(s) local f,e=loadstring(s); return f~=nil,e end")
    for name, data in adapter.sources.items():
        assert parse(data) == (True, None)
        before, after = original.files[name].splitlines(), data.splitlines()
        assert len(before) == len(after)
        region = adapter.report['main_region' if name == script else 'perception_region']
        changed = {i+1 for i, (a,b) in enumerate(zip(before, after)) if a != b}
        allowed = {region['function_line'], region['loop_line'], region['yield_line'], region['function_end']-1}
        if name == script:
            allowed |= {i+1 for i,line in enumerate(before) if b'create_thread("calculation")' in line}
        assert changed <= allowed
        assert hashlib.sha256(original.files[name]).hexdigest() == SOURCES[name]
    modified = SimpleNamespace(files=dict(original.files))
    modified.files['function_command.txt'] += b'\n-- newer revision'
    with pytest.raises(ValueError, match='Unreviewed community'):
        CommunityIntentAdapter(modified, script)


def test_original_perception_reports_exact_unavailable_field_without_fabrication():
    original = package()
    contract = interface()
    teacher = QueryTeacher(contract, spec(PACKAGE) | {
        'script': '01_marisa_main.ai', 'mode': 'intent',
        'source_adapter': 'th123ai_20110925', 'instruction_limit': 1000000})
    episode = teacher.spawn(17)
    warm(contract, episode)
    result = episode.query(observation(contract))
    # Screen-only source initialization needs world x. No synthetic value added.
    assert result.status == 'BLOCKED_INTENT_SOURCE'
    assert 'UNAVAILABLE_STUDENT_FIELD:x' in result.detail
    assert result.action is None and not result.input_legal
    assert teacher.adapter.report['main_region']['loop_line'] == 50
    assert all(hashlib.sha256(original.files[n]).hexdigest() == h for n,h in SOURCES.items())


@pytest.mark.parametrize('helper,expected', [('lvr236', ('236B','214B')),
    ('lvr214', ('214B','236B')), ('lvr623', ('623B','421B')), ('lvr22', ('22B','22B'))])
@pytest.mark.parametrize('facing', [-1,1])
def test_helper_intents_match_original_direction_selection(tmp_path, helper, expected, facing):
    original = package()
    adapter = CommunityIntentAdapter(original, '01_marisa_main.ai')
    for forward, command in zip((True,False), expected):
        # Run only the original helper against ScriptAPI's offline key recorder.
        lua = LuaRuntime(encoding=None, unpack_returned_tuples=True)
        api = ScriptAPI(lua, 17)
        lua.execute(original.api)
        lua.execute(original.files['function_command.txt'])
        lua.globals()[b'my_dir'] = facing
        lua.globals()[b'is_dir_front'] = forward
        step = lua.eval(b"function(f,b) local c=coroutine.create(function() f(b) end); return function() return coroutine.resume(c) end end")(
            lua.globals()[helper.encode()], lua.globals()[b'ACT_B'])
        inputs = []
        for _ in range(12):
            result = step()
            ok = result[0] if isinstance(result, tuple) else result
            if not ok:
                break
            inputs.append(api.inputs())
        horizontal = next((int(row[0])*facing for row in inputs if row[0]), 0)
        assert horizontal == (1 if '6' in command else -1 if '4' in command else 0)
        assert any(row[3] for row in inputs)  # ACT_B is logical channel 3

        directory = tmp_path / f'{helper}-{facing}-{forward}'
        directory.mkdir()
        contract, episode = actor(directory,
            f'function teacher_decide() {helper}(ACT_B) end')
        warm(contract, episode)
        # Install only audited helper interception into a fixture decision source.
        # The original full decision tree is NOT being tested with invented fields.
        episode.env[b'is_dir_front'] = forward
        episode.intents.prepare()
        adapter.install(episode)
        episode.teacher.adapter = SimpleNamespace(install=adapter.install)
        # Avoid full-source lifecycle; this fixture tests just the helper binding.
        episode.intents.initialized = True
        episode.env[b'teacher_observe'] = episode.lua.eval(b'function() end')
        result = episode.query(observation(contract), opportunity=boundary(), boundary_id=2)
        assert result.intent == command
        assert result.status == ('UNMAPPABLE_INTENT' if command == '421B' else 'INTENT_PROPOSED')
        assert 'fixture.ai:' in result.source


def test_unsupported_segment_recovers_at_next_boundary(tmp_path):
    contract, episode = actor(tmp_path, '''
attempts=0
function teacher_decide()
 attempts=attempts+1
 if attempts==1 then wait(30) else command('236B') end
 error('CONTINUATION_MUST_NOT_RUN')
end
''')
    warm(contract, episode)
    first = episode.query(observation(contract), opportunity=boundary(), boundary_id=2)
    assert first.status == 'UNSUPPORTED_INTENT_SEGMENT' and episode.blocked is None
    assert episode.query(observation(contract)).status == 'WAIT_TRUSTED_DECISION_BOUNDARY'
    second = episode.query(observation(contract), opportunity=boundary(), boundary_id=4)
    assert second.intent == '236B' and episode.env[b'attempts'] == 2


def test_original_decision_writes_cannot_commit_an_unexecuted_plan(tmp_path):
    contract, episode = actor(tmp_path, '''
memory={actual=0,assumed=0}; alias=memory
function teacher_observe() memory.actual=memory.actual+1 end
function teacher_decide()
 memory.assumed=99; memory.new={imaginary=true}; RnI_B=1
 command('236B')
end
''')
    warm(contract, episode)
    episode.intents.prepare()
    episode.teacher.adapter = SimpleNamespace(install=lambda _: None, readonly_decision=True)
    episode.intents.initialized = True
    first = episode.query(observation(contract), opportunity=boundary(), boundary_id=2)
    assert first.intent == '236B'
    memory = episode.env[b'memory']
    assert memory[b'actual'] == 1 and memory[b'assumed'] == 0 and memory[b'new'] is None
    assert episode.lua.eval(b'function(e) return rawget(e,"RnI_B"),e.alias==e.memory end')(episode.env) == (None,True)
    episode.intents.reject(first.intent_id)
    episode.query(observation(contract))
    assert memory[b'actual'] == 2 and memory[b'assumed'] == 0
    second = episode.query(observation(contract), opportunity=boundary(), boundary_id=4)
    assert second.intent == '236B' and memory[b'actual'] == 3


def test_snapshot_bound_fails_instead_of_unbounded_copy():
    lua = LuaRuntime(encoding=None)
    table = lua.eval(b'(function() local t={} for i=1,131073 do t[i]=i end return t end)()')
    with pytest.raises(LuaError, match='INTENT_MEMORY_SNAPSHOT_LIMIT'):
        lua.eval(TRANSACTION)(table)

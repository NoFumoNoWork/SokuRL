"""Reviewed, byte-pinned splitting of the 2011 Marisa/Reimu Lua package.

Not a general Lua rewriter or engine-semantics recovery. Conditions and perception
calculations remain original bytes. Only the reviewed function/outer-loop seams
change; unsupported input operations are masked by the query bridge.
"""
import hashlib


SOURCES = {
    '01_marisa_main.ai': '10f53fa1d1c53fc731d4d4862117617d0162c15d6986442dc7f13bea8bad27ae',
    '00_reimu_main.ai': '32c11da92f29135672a8b660d8a6f977622dc4347ebe4caff74ae40f478a6423',
    'def_clc.lua': '732029a309f5b348ef04228f3f19f48c8dac4b47e2f4b5a6c7cbfd9b2ad3fc76',
    'function_command.txt': 'e4f0ed14af9a3b49c78a02a2a1b65852590503336835f9c6d20afee18a87dd8b',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def split_loop(data, function, replacement):
    """Only used after exact source-hash verification; retain original line IDs."""
    lines = data.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.strip() == b'function '+function+b'()')
    loop = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == b'while true do')
    significant = [i for i, line in enumerate(lines) if line.strip()]
    last_yield, loop_end, function_end = significant[-3:]
    if ([lines[i].strip() for i in (last_yield, loop_end, function_end)] !=
            [b'yield();', b'end', b'end']):
        raise ValueError('Pinned source outer-loop suffix changed')
    def replace(index, content):
        newline = b'\r\n' if lines[index].endswith(b'\r\n') else b'\n'
        lines[index] = content + newline
    if function == b'main':
        replace(start, b'function teacher_initialize()')
        registrations = [i for i in range(start, loop) if b'create_thread("calculation")' in lines[i]]
        if len(registrations) != 1:
            raise ValueError('Pinned perception registration changed')
        replace(registrations[0], b'-- perception scheduled explicitly from actual delayed observations')
        replace(loop, b'end; function teacher_decide()')
    else:
        replace(start, b'function '+replacement+b'()')
        replace(loop, b'-- original outer-loop body: exactly one perception update')
    replace(last_yield, b'-- original frame yield removed at the reviewed outer-loop boundary')
    replace(loop_end, b'-- original outer-loop end removed')
    return b''.join(lines), dict(function_line=start+1, loop_line=loop+1,
                                yield_line=last_yield+1, function_end=function_end+1)


HANDLERS = b"""
function(env)
  local info = debug.getinfo
  local function button(n)
    if n == env.ACT_A then return 'A' end
    if n == env.ACT_B then return 'B' end
    if n == env.ACT_C then return 'C' end
    return nil
  end
  for name,motions in pairs({lvr236={'236','214'}, lvr214={'214','236'},
                             lvr623={'623','421'}, lvr22={'22','22'}}) do
    local choices, helper = motions, name
    env[name] = function(n)
      local suffix = button(n)
      if not suffix then return env.unsupported_intent(helper .. ':button') end
      local forward = env.is_dir_front
      if type(forward) ~= 'boolean' then error('INTENT_DIRECTION_UNAVAILABLE') end
      local at = info(2, 'Sl')
      return env.propose_intent(choices[forward and 1 or 2] .. suffix,
        at.source .. ':' .. tostring(at.currentline))
    end
  end
  -- This source reset only clears planned keys when no held B/C is pending.
  -- A held branch cannot be treated as a fresh command; mask, do not mutate it.
  env.key_resetEX = function()
    if env.RnI_B ~= 0 or env.RnI_C ~= 0 then
      return env.unsupported_intent('key_resetEX:held-shot')
    end
  end
end
"""


class CommunityIntentAdapter:
    readonly_decision = True

    def __init__(self, package, script):
        if script not in ('01_marisa_main.ai', '00_reimu_main.ai'):
            raise ValueError('Source adapter supports only the two reviewed 2011 main scripts')
        for name, expected in SOURCES.items():
            if name not in package.files or digest(package.files[name]) != expected:
                raise ValueError(f'Unreviewed community source version: {name}')
        main, main_region = split_loop(package.files[script], b'main', b'teacher_decide')
        perception, perception_region = split_loop(package.files['def_clc.lua'], b'calculation', b'teacher_observe')
        self.sources = {script: main, 'def_clc.lua': perception}
        self.report = dict(adapter='th123ai_20110925', original_sources=SOURCES,
            generated_sha256={k:digest(v) for k,v in self.sources.items()},
            main_region=main_region, perception_region=perception_region,
            evidence='PINNED_SOURCE_ADAPTATION_NOT_GAMEPLAY_OR_ORIGINAL_POLICY_EQUIVALENCE',
            scheduling='initialize once; original perception body once per delayed query; fresh decision at supplied boundary',
            differences=['perception-before-decision differs from original thread scheduling',
                'post-proposal continuation discarded, not assumed executed',
                'decision global/table writes rolled back; only observation-phase memory commits',
                'original decisions with strategic writes need review before those writes can be retained',
                'global/table transaction is not a general Lua closure-upvalue transaction',
                'raw keys/waits/held variants masked; missing observation fields BLOCKED'],
            helpers={'lvr236':['236','214'], 'lvr214':['214','236'],
                     'lvr623':['623','421 (unrepresentable in current catalog)'], 'lvr22':['22','22']})

    def install(self, episode):
        episode.lua.eval(HANDLERS)(episode.env)


TRANSACTION = b"""
function(env)
  local seen, saved, entries = {}, {}, 0
  local function visit(t)
    if seen[t] then return end
    seen[t] = true
    local copy = {}; saved[t] = copy
    for k,v in pairs(t) do
      entries = entries+1
      if entries > 131072 then error('INTENT_MEMORY_SNAPSHOT_LIMIT') end
      copy[k] = v
      if type(k)=='table' then visit(k) end
      if type(v)=='table' then visit(v) end
    end
  end
  visit(env)
  return function()
    for t,copy in pairs(saved) do
      local remove = {}
      for k in pairs(t) do if copy[k]==nil then remove[#remove+1]=k end end
      for _,k in ipairs(remove) do rawset(t,k,nil) end
      for k,v in pairs(copy) do rawset(t,k,v) end
    end
  end
end
"""

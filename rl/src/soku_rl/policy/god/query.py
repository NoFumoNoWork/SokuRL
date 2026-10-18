"""Fail-closed Lua advice over the student's existing animation observation.

This is not GodPolicy's privileged adapter. Missing world coordinates, timers,
enemy inputs and boxes remain missing; Lua may not read the game or filesystem.
"""
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from lupa.lua51 import LuaRuntime, LuaError, LuaMemoryError

from soku_rl.env.encoding import encode_action
from soku_rl.env.observation.animation import frame_width, HISTORY_OFFSETS
from .api import ScriptAPI
from .package import ScriptPackage
from .runtime import SCHEDULER


@dataclass(frozen=True)
class Advice:
    status: str
    action: int | None = None
    command: int | None = None
    source: str = ''
    detail: str = ''
    input_legal: bool = False
    engine_acceptance: str = 'NOT_OBSERVED'

    def record(self):
        return asdict(self)


BOOTSTRAP = b"""
local native_load, native_setfenv = loadstring, setfenv
local hook, info, gethook = debug.sethook, debug.getinfo, debug.gethook
local remaining, last, blocked_at = 0, '', nil
local function trace(event, line)
  remaining = remaining - (event == 'count' and 100 or 1)
  if remaining <= 0 then error('QUERY_INSTRUCTION_LIMIT') end
  if event ~= 'count' then
    local at = info(2, 'S')
    last = at.source .. ':' .. tostring(line)
  end
end
local env = {}
for _, name in ipairs({'assert','error','ipairs','next','pairs','pcall','select',
  'tonumber','tostring','type','unpack','xpcall'}) do env[name] = _G[name] end
for _, name in ipairs({'math','table','string'}) do
  env[name] = {}; for k,v in pairs(_G[name]) do env[name][k] = v end
end
env.coroutine = {}
for k,v in pairs(coroutine) do env.coroutine[k] = v end
local create = coroutine.create
env.coroutine.create = function(f)
  local thread = create(f); hook(thread, trace, 'l', 100); return thread
end
env._G = env
env.print = function(...) end
env._script_error = function(message) error(message) end
setmetatable(env, {__index=function(_, key)
  local at = info(2, 'Sl')
  blocked_at = at.source .. ':' .. tostring(at.currentline)
  error('UNAVAILABLE_STUDENT_FIELD:' .. tostring(key))
end})
local function execute(bytes, name)
  local f, message = native_load(bytes, name)
  if not f then error(message) end
  native_setfenv(f, env)
  local previous, mask, count = gethook()
  hook(trace, 'l', 100)
  local result = {pcall(f)}
  hook(previous, mask, count)
  if not result[1] then error(result[2]) end
  return unpack(result, 2)
end
return env, execute, function(n) remaining=n; last=''; blocked_at=nil end,
  function() return blocked_at or last end
"""


def validate_spec(spec):
    keys = {'package', 'api_source', 'script', 'reaction_frames', 'instruction_limit', 'seed'}
    if not isinstance(spec, dict) or set(spec) != keys:
        raise ValueError('expert requires explicit package/API/script, reaction_frames, instruction_limit and seed')
    if any(not isinstance(spec[k], str) or not spec[k] for k in ('package', 'api_source', 'script')):
        raise ValueError('expert source paths and script must be nonempty')
    if type(spec['reaction_frames']) is not int or spec['reaction_frames'] < 1:
        raise ValueError('expert reaction_frames must be at least one, not instant reaction')
    if type(spec['instruction_limit']) is not int or not 100 <= spec['instruction_limit'] <= 1_000_000:
        raise ValueError('expert instruction_limit must be bounded')
    if type(spec['seed']) is not int or not 0 <= spec['seed'] < 2**32:
        raise ValueError('expert seed must be uint32')


class QueryTeacher:
    def __init__(self, interface, spec):
        validate_spec(spec)
        episode = interface.episode
        if (episode.observation_mode != 'animation' or episode.history_frames != len(HISTORY_OFFSETS)
                or (episode.decision_frames, episode.latency_frames) != (1, 0)
                or interface.action_space.n != 90 or interface.config.action_history < 1):
            raise ValueError('restricted expert requires existing 1F/90-action animation/history contract')
        self.interface, self.spec = interface, dict(spec)
        self.package = ScriptPackage(spec['package'], spec['api_source'])
        self.package.source(spec['script'])
        if self.package.repairs:
            # Do not silently apply the privileged opponent's documented repairs.
            self.repairs = tuple(sorted(self.package.repairs))
        else:
            self.repairs = ()
        self.fingerprint = hashlib.sha256(json.dumps({
            'package': self.package.original_fingerprint, 'spec': spec,
            'implementation': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('query.py', 'api.py', 'runtime.py', 'package.py')},
            'projection': 'student_animation_only_v1', 'home_advantage': 'disabled',
        }, sort_keys=True).encode()).hexdigest()

    def spawn(self, seed):
        return QueryEpisode(self, seed)


class QueryEpisode:
    def __init__(self, teacher, seed):
        self.teacher, self.frame = teacher, -1
        self.history = deque(maxlen=teacher.spec['reaction_frames'] + 1)
        self.blocked = None
        # Python attributes, eval/builtins and Lua system libraries are not exposed.
        self.lua = LuaRuntime(encoding=None, unpack_returned_tuples=True,
            max_memory=64 * 1024 * 1024,
            register_eval=False, register_builtins=False,
            attribute_filter=lambda *args: (_ for _ in ()).throw(AttributeError('Python access denied')))
        self.api = ScriptAPI(self.lua, seed)
        self.env, self.execute, self.budget, self.location = self.lua.execute(BOOTSTRAP)
        for name in ('key_on', 'key_off', 'key_reset', 'get_key_stat', 'get_key_map'):
            self.env[name.encode()] = getattr(self.api, name)
        self.env[b'getfunc'] = self.lua.eval(b"function(env) return function(f) if type(f)=='string' then return env[f] else return f end end end")(self.env)
        self.env[b'require'] = self.require
        self.budget(teacher.spec['instruction_limit'])
        # BOOTSTRAP copied ScriptAPI's existing actor-local random implementation.
        self.env[b'math'][b'randomseed'] = self.api.randomseed
        self.execute(teacher.package.api, b'@community/api.ai')
        self.execute(SCHEDULER, b'@query/scheduler')
        self.started = False
        self.previous_command = None

    def require(self, name):
        package = self.teacher.package
        alias = package.resolve_name(name)
        relative = package.aliases[alias]
        # Original bytes only, even if the old privileged adapter has a repair.
        return self.execute(package.files[relative], b'@community/' + relative.encode('utf-8'))

    def query(self, observation):
        interface = self.teacher.interface
        if (observation.shape != interface.observation_space.shape
                or observation.dtype != np.float32 or not np.isfinite(observation).all()):
            raise ValueError('expert query must receive the unmodified student observation')
        self.frame += 1
        self.history.append(observation.copy())
        if self.blocked is not None:
            return self.blocked
        if self.frame < self.teacher.spec['reaction_frames']:
            return Advice('REACTION_WARMUP')
        old = self.history[0]
        width = frame_width(interface.episode.animation_object_slots)
        current = interface.base_observation(old).reshape(-1, width)[-1]
        if not (current[0] and current[10]):
            return Advice('UNAVAILABLE_VISIBLE_FIGHTER')
        for offset, prefix in ((0, b'my_'), (10, b'enemy_')):
            token = current[offset:offset+10]
            for name, value in ((b'char', int(token[4])), (b'act', int(token[5])),
                                (b'dir', int(token[3]))):
                self.env[prefix+name] = value
        # Explicit neutral-context intervention; no stage/BGM information is read.
        self.env[b'stage_number'], self.env[b'bgm_number'] = -1, -1
        # Teacher key state is an intention, not a claim it was executed. The
        # learner's own most recent submitted input is available without delay.
        # Do not supervise continuations of a prefix the student did not execute.
        previous_executed = encode_action(tuple(map(int, observation[-8:])))
        aligned = self.previous_command is None or self.previous_command == previous_executed
        try:
            self.budget(self.teacher.spec['instruction_limit'])
            if not self.started:
                self.require(self.teacher.spec['script'])
                self.step = self.env[b'make_step']()
                self.started = True
            self.step()
            command = encode_action(self.api.inputs())
        except (LuaError, LuaMemoryError, ValueError, FileNotFoundError, AttributeError) as error:
            # An exception from a main-thread hook can bypass its Lua cleanup.
            # Clear it through the native C entry before asking Lua for evidence.
            self.lua.globals()[b'debug'][b'sethook']()
            self.blocked = Advice('BLOCKED_SOURCE_DEPENDENCY', source=self.location().decode('utf-8', 'replace'), detail=str(error))
            return self.blocked
        self.previous_command = command
        if not aligned:
            return Advice('TEACHER_PREFIX_NOT_EXECUTED', command=command,
                          source=self.location().decode('utf-8', 'replace'))
        if command & 48:
            return Advice('MASKED_CARD_INPUT', command=command, source=self.location().decode('utf-8', 'replace'))
        if command not in interface.inverse:
            return Advice('UNREPRESENTABLE_INPUT', command=command, source=self.location().decode('utf-8', 'replace'))
        return Advice('VALID_INPUT', interface.action(command), command,
            source=self.location().decode('utf-8', 'replace'), input_legal=True)

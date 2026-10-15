"""Resume unchanged Lua strategies at each original _yield frame boundary."""
import hashlib
from functools import lru_cache

from lupa.lua51 import LuaRuntime, LuaSyntaxError

from soku_rl.env.encoding import encode_action
from soku_rl.env.observation.memory_schema import PRIVILEGED_FEATURES
from soku_rl.env.observation.privileged import PrivilegedObservation, decode_privileged
from soku_rl.policy.base import PrivilegedPlayActor, RulePolicy
from .api import ScriptAPI
from .package import ScriptPackage
from .random_logs import install_random_logs


SCHEDULER = b"""
local frame_marker = {}
function _yield() coroutine.yield(frame_marker) end
function thread_main()
  while thread_num >= 2 do
    local i = 0
    while i < thread_num do
      local args = thread_param[i]
      while true do
        local result = {coroutine.resume(thread_list[i], unpack(args, 1))}
        args = {}
        thread_param[i] = {}
        if not result[1] then
          _script_error(result[2])
          table.remove(thread_list, i)
          table.remove(thread_param, i)
          thread_num = thread_num - 1
          i = i - 1
          break
        end
        if result[2] == frame_marker then
          coroutine.yield(frame_marker)
        else
          break
        end
      end
      i = i + 1
    end
  end
end
function make_step()
  local runner = coroutine.create(api_main)
  return function()
    local ok, value = coroutine.resume(runner)
    if not ok then error(value) end
    if value ~= frame_marker then error('script stopped before its next frame') end
  end
end
"""


@lru_cache(maxsize=512)
def compiled_source(source, name):
    """Cache immutable bytecode; never share a Lua state between actors."""
    compiler = LuaRuntime(encoding=None)
    try:
        return compiler.eval(b"string.dump")(compiler.compile(source, name=name))
    except LuaSyntaxError as error:
        # The original package also contains data and documentation. An invalid
        # chunk is an error only when selected or required as executable code.
        return str(error)


class GodPolicy(RulePolicy):
    def __init__(self, name, package, script, episode):
        if episode.observation_mode != "privileged_state" or (episode.decision_frames, episode.latency_frames) != (1, 0):
            raise ValueError("original community scripts require complete state and frame-by-frame control")
        self.name, self.package, self.script, self.episode = name, package, script, episode
        self.programs = {alias: compiled_source(package.source(alias), b"@original/"+alias)
                         for alias in package.aliases}
        self.api_program = compiled_source(package.api, b"@original/api.ai")
        self.scheduler_program = compiled_source(SCHEDULER, b"@sokurl/scheduler")
        if isinstance(self.api_program, str) or isinstance(self.scheduler_program, str):
            raise LuaSyntaxError("invalid original API or frame scheduler")
        if script != "character":
            self.program(script)
        self.fingerprint = hashlib.sha256((package.fingerprint + script).encode()).hexdigest()

    def spawn(self, seed):
        return GodActor(self, seed)

    def program(self, name):
        code = self.programs[self.package.resolve_name(name)]
        if isinstance(code, str):
            raise LuaSyntaxError(code)
        return code

    def spawn_play(self, seed):
        # The original host reloads on battle scene entry, not on each knockout.
        return PrivilegedPlayActor(self.spawn(seed), False)


class GodActor:
    def __init__(self, policy, seed):
        self.policy = policy
        self.lua = LuaRuntime(encoding=None, unpack_returned_tuples=True)
        self.random_logs = install_random_logs(self.lua)
        self.api = ScriptAPI(self.lua, seed)
        self.logs = []
        self.failures = []
        self.lua.globals()[b"_script_error"] = self.failures.append
        self.lua.globals()[b"print"] = self.record_message
        self.lua.globals()[b"require"] = self.require
        self.lua.execute(policy.api_program)
        self.lua.execute(policy.scheduler_program)
        self.started = False
        self.last_frame = -1

    def record_message(self, *parts):
        if len(self.logs) < 100:
            self.logs.append(tuple(parts))

    def require(self, name):
        self.lua.execute(self.policy.program(name))

    def act(self, observation):
        expected = PRIVILEGED_FEATURES * self.policy.episode.history_frames
        if observation.shape != (expected,):
            raise ValueError("original script observation has incorrect shape")
        current = decode_privileged(observation[-PRIVILEGED_FEATURES:])
        return self.act_observation(current)

    def act_observation(self, current):
        if not isinstance(current, PrivilegedObservation) or len(current.players) != 2:
            raise TypeError("original scripts require a complete decoded observation")
        frame = int(current.world["frame"])
        if frame != self.last_frame + 1:
            raise ValueError("original scripts require every consecutive simulation frame")
        self.api.observe(current)
        if not self.started:
            script = (self.policy.package.character_script(int(current.players[0]["char"]))
                      if self.policy.script == "character" else self.policy.script)
            self.require(script.encode("utf-8"))
            self.step = self.lua.globals()[b"make_step"]()
            self.started = True
        self.step()
        if self.failures:
            raise RuntimeError(f"original strategy failed: {self.failures[-1]!r}")
        self.last_frame = frame
        return encode_action(self.api.inputs())

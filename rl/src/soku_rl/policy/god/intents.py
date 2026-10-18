"""Single-decision Lua advice, never an unexecuted input coroutine continuation.

A reviewed source adapter must expose teacher_decide(). Its globals/upvalues live
for the episode; each invocation is disposable. This does not rewrite main().
"""
from lupa.lua51 import LuaError, LuaMemoryError

from soku_rl.policy.hierarchical_control import Opportunity, command_catalog


BRIDGE = b"""
function(env)
  local info, raw = debug.getinfo, rawget
  local function propose(command, source)
    local at = info(2, 'Sl')
    return env.coroutine.yield({kind='INTENT', command=command,
      source=source or at.source .. ':' .. tostring(at.currentline)})
  end
  local function denied(name)
    return function()
      local at = info(2, 'Sl')
      return env.coroutine.yield({kind='UNSUPPORTED_OPERATION', operation=name,
        source=at.source .. ':' .. tostring(at.currentline)})
    end
  end
  return function()
    env.propose_intent = propose
    env.command = propose
    env.unsupported_intent = function(name) return denied(name)() end
    for _,name in ipairs({'key_on','key_off','key_reset','get_key_stat',
      'get_key_map','create_thread','wait','yield'}) do env[name] = denied(name) end
  end, function() return raw(env, 'teacher_decide') end,
  function(f, complete)
    local thread = env.coroutine.create(f)
    local ok, value = env.coroutine.resume(thread)
    if not ok then error(value) end
    if complete and env.coroutine.status(thread) ~= 'dead' then
      error('INTENT_PERCEPTION_OR_INITIALIZATION_MUST_COMPLETE')
    end
    -- The thread is never returned or resumed, including after an accepted intent.
    return value
  end
end
"""


class IntentSession:
    def __init__(self, episode):
        self.episode = episode
        self.catalog = command_catalog()
        self.indices = {row['name']: i for i, row in enumerate(self.catalog)}
        self.install, self.entry, self.invoke = episode.lua.eval(BRIDGE)(episode.env)
        self.loaded = False
        self.initialized = False
        self.last_boundary = -1
        self.serial = 0
        self.pending = None
        self.invalidations = 0

    def prepare(self):
        episode = self.episode
        self.install()
        if not self.loaded:
            episode.require(episode.teacher.spec['script'])
            self.loaded = True
            self.install()
        adapter = episode.teacher.adapter
        if adapter is not None:
            adapter.install(episode)
            if not self.initialized:
                self.invoke(episode.env[b'teacher_initialize'], True)
                self.initialized = True

    def source_error(self, error):
        from .query import Advice
        episode = self.episode
        episode.lua.globals()[b'debug'][b'sethook']()
        episode.blocked = Advice('BLOCKED_INTENT_SOURCE',
            source=episode.location().decode('utf-8', 'replace'), detail=str(error))
        return episode.blocked

    def reject(self, intent_id):
        """Invalidate only this proposal, not the episode's tactical Lua memory."""
        if self.pending is None or intent_id != self.pending.intent_id:
            raise ValueError('reject requires the currently pending intent identity')
        self.pending = None
        self.invalidations += 1

    def query(self, opportunity, boundary_id):
        from .query import Advice

        episode = self.episode
        # Perception follows the real delayed observation clock, even when no
        # choice is offered. It never advances a teacher-only action coroutine.
        if episode.teacher.adapter is not None:
            try:
                episode.budget(episode.teacher.spec['instruction_limit'])
                self.prepare()
                self.invoke(episode.env[b'teacher_observe'], True)
            except (LuaError, LuaMemoryError, ValueError, FileNotFoundError,
                    AttributeError, TypeError) as error:
                return self.source_error(error)
        if (not isinstance(opportunity, Opportunity) or
                opportunity.legality_source != 'PROVIDED_ENGINE_LEGAL_SET' or
                not opportunity.can_respond or opportunity.forced):
            return Advice('WAIT_TRUSTED_DECISION_BOUNDARY')
        if (type(boundary_id) is not int or boundary_id != episode.frame or
                boundary_id <= self.last_boundary):
            return Advice('STALE_DECISION_BOUNDARY')
        if (len(opportunity.legal_actions) != len(self.catalog) or
                any(type(x) is not bool for x in opportunity.legal_actions)):
            raise ValueError('intent opportunity must use the existing command catalog')
        self.last_boundary = boundary_id
        # A new boundary always reassesses the actual observed world. Never finish
        # the preceding coroutine or inject its keys, even if the intent repeats.
        self.pending = None
        try:
            if episode.teacher.adapter is None:
                episode.budget(episode.teacher.spec['instruction_limit'])
            self.prepare()
            entry = self.entry()
            if entry is None:
                episode.blocked = Advice('BLOCKED_INTENT_ADAPTER',
                    detail='Reviewed teacher_decide entry missing; original main is not restarted')
                return episode.blocked
            restore = None
            if getattr(episode.teacher.adapter, 'readonly_decision', False):
                from .source_adapter import TRANSACTION
                restore = episode.lua.eval(TRANSACTION)(episode.env)
            try:
                value = self.invoke(entry)
            finally:
                if restore is not None:
                    restore()
            if value is not None and hasattr(value, '__getitem__') and value[b'kind'] == b'UNSUPPORTED_OPERATION':
                return Advice('UNSUPPORTED_INTENT_SEGMENT',
                    source=value[b'source'].decode('utf-8', 'replace'),
                    detail=value[b'operation'].decode('utf-8', 'replace'),
                    decision_frame=boundary_id)
            if (value is None or not hasattr(value, '__getitem__') or value[b'kind'] != b'INTENT'):
                return Advice('NO_INTENT', decision_frame=boundary_id)
            command = value[b'command']
            source = value[b'source'].decode('utf-8', 'replace')
            if not isinstance(command, bytes):
                return Advice('UNMAPPABLE_INTENT', source=source, decision_frame=boundary_id)
            name = command.decode('ascii', 'strict')
            index = self.indices.get(name)
            if index is None:
                return Advice('UNMAPPABLE_INTENT', intent=name, source=source,
                    decision_frame=boundary_id)
            if not opportunity.legal_actions[index]:
                return Advice('MASKED_INTENT', intent=name, source=source,
                    decision_frame=boundary_id)
            self.serial += 1
            self.pending = Advice('INTENT_PROPOSED', intent=name, intent_id=self.serial,
                source=source, decision_frame=boundary_id,
                detail='Proposal only; no native acceptance or 90-action label asserted')
            return self.pending
        except (LuaError, LuaMemoryError, ValueError, FileNotFoundError,
                AttributeError, UnicodeError, TypeError) as error:
            return self.source_error(error)

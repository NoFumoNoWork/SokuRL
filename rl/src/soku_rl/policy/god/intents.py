"""Single-decision Lua advice, never an unexecuted input coroutine continuation.

A reviewed source adapter must expose teacher_decide(). Its globals/upvalues live
for the episode; each invocation is disposable. This does not rewrite main().
"""
from lupa.lua51 import LuaError, LuaMemoryError

from soku_rl.policy.hierarchical_control import Opportunity, command_catalog


BRIDGE = b"""
function(env)
  local info, raw = debug.getinfo, rawget
  local function propose(command)
    local at = info(2, 'Sl')
    return env.coroutine.yield({kind='INTENT', command=command,
      source=at.source .. ':' .. tostring(at.currentline)})
  end
  local function denied(name)
    return function() error('INTENT_ADAPTER_REQUIRED:' .. name) end
  end
  return function()
    env.propose_intent = propose
    env.command = propose
    for _,name in ipairs({'key_on','key_off','key_reset','get_key_stat',
      'get_key_map','create_thread','wait','yield'}) do env[name] = denied(name) end
  end, function() return raw(env, 'teacher_decide') end,
  function(f)
    local thread = env.coroutine.create(f)
    local ok, value = env.coroutine.resume(thread)
    if not ok then error(value) end
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
        self.last_boundary = -1
        self.serial = 0
        self.pending = None
        self.invalidations = 0

    def reject(self, intent_id):
        """Invalidate only this proposal, not the episode's tactical Lua memory."""
        if self.pending is None or intent_id != self.pending.intent_id:
            raise ValueError('reject requires the currently pending intent identity')
        self.pending = None
        self.invalidations += 1

    def query(self, opportunity, boundary_id):
        from .query import Advice

        episode = self.episode
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
            episode.budget(episode.teacher.spec['instruction_limit'])
            self.install()
            if not self.loaded:
                episode.require(episode.teacher.spec['script'])
                self.loaded = True
                self.install()
            entry = self.entry()
            if entry is None:
                episode.blocked = Advice('BLOCKED_INTENT_ADAPTER',
                    detail='Reviewed teacher_decide entry missing; original main is not restarted')
                return episode.blocked
            value = self.invoke(entry)
            if (value is None or not hasattr(value, '__getitem__') or
                    value[b'kind'] != b'INTENT'):
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
            episode.lua.globals()[b'debug'][b'sethook']()
            episode.blocked = Advice('BLOCKED_INTENT_SOURCE',
                source=episode.location().decode('utf-8', 'replace'), detail=str(error))
            return episode.blocked

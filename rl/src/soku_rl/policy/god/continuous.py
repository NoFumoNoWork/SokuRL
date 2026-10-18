"""Recoverable source plans over supplied observations, never a game controller.

The caller owns input-consumption and legal-opportunity evidence. No native
reader, event decoder, fallback labels, or PPO attachment is provided here.
"""
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum

from lupa.lua51 import LuaRuntime, LuaError

from soku_rl.env.encoding import decode_action, encode_action
from soku_rl.env.observation.shared_native import decode_shared
from .shared_native import SharedNativeAPI
from .source_adapter import SOURCES, digest, split_loop
from .scoped_state import API_SHA256


class PlanState(str, Enum):
    READY = 'READY'
    ACTIVE = 'ACTIVE'
    INVALIDATED = 'INVALIDATED'
    WAITING = 'WAITING'


@dataclass(frozen=True)
class ExecutionReceipt:
    """One completed frame, supplied by a separately validated native adapter.

    Synthetic fixtures must not be described as native proof. accepted is an
    independently attributed native action acceptance, NOT key equality.
    """
    frame: int
    action: int
    facing: int
    poll_index: int | None
    consumed: bool
    accepted: bool = False
    interrupted: bool = False
    accepted_plan_id: int | None = None

    def __post_init__(self):
        if (type(self.frame) is not int or self.frame < 0 or
                type(self.action) is not int or not 0 <= self.action < 90 or
                type(self.facing) is not int or self.facing not in (-1, 1) or
                (self.poll_index is not None and (type(self.poll_index) is not int or self.poll_index < 0)) or
                any(type(v) is not bool for v in (self.consumed, self.accepted, self.interrupted)) or
                (self.accepted and (not self.consumed or type(self.accepted_plan_id) is not int
                                    or self.accepted_plan_id <= 0))):
            raise ValueError('invalid execution receipt')


SCOPE = br'''function(public)
 local mode, memory, plan = 'init', {}, {}
 local protected=setmetatable({}, {__mode='k'})
 local function readonly(t)
   if protected[t] then return protected[t] end
   local p={};protected[t]=p
   setmetatable(p,{__index=function(_,k)
     local v=t[k];if type(v)=='table' then return readonly(v) end;return v
   end,__newindex=function() error('PLAN_MUTATES_OBSERVED_TABLE') end})
   return p
 end
 local env;env=setmetatable({}, {__index=function(_,k)
   if k=='_G' then return env end
   if mode=='decision' and plan[k]~=nil then return plan[k] end
   local v=memory[k];if v==nil then v=public[k] end
   if v==nil then error('UNTRUSTED_SOURCE_DEPENDENCY:'..tostring(k)) end
   if mode=='decision' and type(v)=='table' then return readonly(v) end
   return v
 end,__newindex=function(_,k,v)
   if mode=='decision' then plan[k]=v else memory[k]=v end
 end})
 return env,function(m) mode=m end,function() plan={} end,memory,
   function(code,name)
     local f,e=loadstring(code,name);if not f then error(e) end
     setfenv(f,env);return f
   end,
   function(f)
     local t=coroutine.create(f)
     return function()
       local ok,v=coroutine.resume(t);if not ok then error(v) end
       return coroutine.status(t)=='dead',v
     end
   end
end'''


class ContinuousSourceTeacher:
    """Pinned original perception, disposable main execution scopes.

    This deliberately changes original thread ordering to observe-before-decide.
    Main-only tactical writes are plans, not confirmed persistent game facts.
    """
    def __init__(self, package, script, seed, action_interface, *, initial_action,
                 instruction_limit=200000):
        if script not in ('01_marisa_main.ai', '00_reimu_main.ai'):
            raise ValueError('reviewed Reimu/Marisa source required')
        if digest(package.api) != API_SHA256 or any(
                digest(package.files[name]) != sha for name, sha in SOURCES.items()):
            raise ValueError('unreviewed source identity')
        self.package, self.script = package, script
        if len(action_interface.commands) != 90:
            raise ValueError('existing combat LearningInterface required')
        self.interface = action_interface
        self.interface.command(initial_action)
        self.initial_action = initial_action
        self.limit = instruction_limit
        self.lua = LuaRuntime(encoding=None, unpack_returned_tuples=True,
            register_eval=False, register_builtins=False)
        self.api = SharedNativeAPI(self.lua, seed)
        self.lua.execute(package.api)
        (self.env, self.mode, self.clear_plan, self.memory,
         self.compile, self.coroutine) = self.lua.eval(SCOPE)(self.lua.globals())
        self.env[b'require'] = self.require
        self.env[b'print'] = lambda *args: None
        self.loaded = set()
        self.require(script)
        # Explicit frame marker and call-site provenance; no original scheduler
        # is resumed after invalidation. wait/key_event retain original API code.
        self.env[b'yield'] = self.lua.eval(br'''function()
          local at=debug.getinfo(2,'Sl');local helpers={}
          for i=2,12 do
            local f=debug.getinfo(i,'n');if not f then break end
            if f.name then helpers[#helpers+1]=f.name end
          end
          coroutine.yield({source=at.source,line=at.currentline,helpers=helpers})
        end''')
        # API functions were defined outside our scoped chunk. Rebind only these
        # Lua helpers, not Python callbacks or standard-library functions.
        for name in (b'wait', b'key_event', b'key_event2'):
            self.lua.eval(b'setfenv')(self.env[name], self.env)
        self.frame = -1
        self.state = PlanState.READY
        self.runner = None
        self.pending = None
        self.last_poll = None
        self.serial = 0
        self.counts = Counter()
        self.reasons = Counter()
        self.mask_reasons = Counter()
        self.trace = deque(maxlen=2048)
        self.invalidated_at = None
        self.initialized = False
        self.fatal = None
        self.last_submitted = None
        self.last_reconciliation = None
        self.budget = self.lua.eval(br'''function(n)
          debug.sethook(function() error('TEACHER_INSTRUCTION_BUDGET') end,'',n)
        end''')
        self.unhook = self.lua.eval(b'function() debug.sethook() end')

    def require(self, name):
        name = name.decode('ascii') if isinstance(name, bytes) else name
        if name not in SOURCES:
            raise ValueError('unreviewed dependency')
        if name in self.loaded:
            return
        code = self.package.files[name]
        if name == self.script:
            code, _ = split_loop(code, b'main', b'teacher_decide')
        elif name == 'def_clc.lua':
            code, _ = split_loop(code, b'calculation', b'teacher_observe')
        self.loaded.add(name)
        self.compile(code, ('@community/' + name).encode())()

    def call(self, function):
        self.budget(self.limit)
        try:
            return function()
        finally:
            self.unhook()

    def invalidate(self, reason):
        self.runner = self.pending = None
        self.clear_plan()
        self.api.events.clear()
        self.state = PlanState.INVALIDATED
        self.invalidated_at = self.frame
        self.counts['invalidations'] += 1
        self.reasons[reason] += 1

    def reconcile(self, receipt, facing):
        if self.pending is None:
            return
        p = self.pending
        self.last_reconciliation = dict(plan_id=self.serial, proposed=p['candidate'],
            actual=None if receipt is None else receipt.action,
            poll_index=None if receipt is None else receipt.poll_index,
            stage='INTENT', reason=None)
        reason = None
        if receipt is None or receipt.frame != self.frame - 1:
            reason = 'MISSING_FRAME_RECEIPT'
        elif (receipt.facing not in (-1, 1) or receipt.facing != p['facing']
              or facing != p['facing']):
            reason = 'FACING_CHANGED'
        elif receipt.interrupted:
            reason = 'NATIVE_INTERRUPT'
        elif not receipt.consumed or type(receipt.poll_index) is not int:
            reason = 'INPUT_CONSUMPTION_UNPROVEN'
        elif self.last_poll is not None and receipt.poll_index != self.last_poll + 1:
            reason = 'INPUT_POLL_GAP'
        elif receipt.action != p['candidate']:
            reason = 'STUDENT_DIVERGED'
        elif receipt.accepted and receipt.accepted_plan_id != self.serial:
            reason = 'FOREIGN_NATIVE_ACCEPTANCE'
        else:
            self.last_poll = receipt.poll_index
            self.last_submitted = receipt.action
            self.counts['submitted'] += 1
            self.last_reconciliation['stage'] = 'SUBMITTED'
            if receipt.accepted:
                self.counts['accepted'] += 1
                self.last_reconciliation['stage'] = 'ACCEPTED'
            # An attack key does not prove that the original action started.
            if any(decode_action(self.interface.command(receipt.action)).inputs[2:5]) and not receipt.accepted:
                reason = 'ACTION_ACCEPTANCE_UNPROVEN'
        if reason:
            self.last_reconciliation['reason'] = reason
            self.invalidate(reason)
        else:
            self.pending = None
            if p['done']:
                self.runner = None
                self.clear_plan()
                self.state = PlanState.READY

    def query(self, values, *, schema, receipt=None, legal_actions=None,
              trusted_boundary=False, replan=False):
        """legal_actions is a caller-verified 90-input mask, not our move table."""
        current = decode_shared(values, schema=schema)
        frame = int(current.world['frame'])
        if frame != self.frame + 1:
            raise ValueError('consecutive frames from episode start required; create new teacher at Reset')
        expected = 1 if self.script.startswith('01_') else 0
        if current.players[0]['char'] != expected:
            raise ValueError('source/player mismatch')
        if legal_actions is not None and (len(legal_actions) != 90 or
                any(type(v) is not bool for v in legal_actions)):
            raise ValueError('explicit 90-action boolean mask required')
        if receipt is not None and (not isinstance(receipt, ExecutionReceipt) or receipt.frame != frame - 1):
            raise ValueError('receipt must describe the immediately preceding frame')
        if type(trusted_boundary) is not bool or type(replan) is not bool or (replan and not trusted_boundary):
            raise ValueError('replanning requires an explicit trusted boundary')
        self.frame = frame
        self.counts['queries'] += 1
        self.api.observe(current)
        # Version1 has no additional reaction delay. Never silently flush or
        # reinterpret a nonzero delay as if its queued inputs had been consumed.
        if self.api.delays[b'key_delay'] or self.api.delays[b'data_delay']:
            self.invalidate('UNSUPPORTED_DELAY_QUEUE')
            return self.result(None, 'UNSUPPORTED_DELAY_QUEUE')
        facing = int(current.players[0]['dir'])
        if facing not in (-1, 1):
            raise ValueError('valid current facing required')
        self.last_reconciliation = None
        self.reconcile(receipt, facing)
        if replan and self.runner is not None:
            self.invalidate('NATIVE_REPLAN_BOUNDARY')
        if self.fatal:
            return self.result(None, 'RESET_REQUIRED_AFTER_PARTIAL_OBSERVATION_FAILURE')
        self.mode(b'observation')
        try:
            if not self.initialized:
                self.call(self.env[b'teacher_initialize'])
                self.initialized = True
            self.call(self.env[b'teacher_observe'])
            # Reviewed observation-only Reimu resets/confirmed spell flags.
            if self.script.startswith('00_'):
                lines = self.package.files[self.script].splitlines(keepends=True)
                self.call(self.compile(b'\n' * 52 + b''.join(lines[52:63]),
                    b'@community/00_reimu_main.ai#observed_flags'))
            self.counts['observations'] += 1
        except (LuaError, ValueError, TypeError, KeyError) as error:
            self.fatal = str(error)
            self.invalidate('OBSERVATION_DEPENDENCY_FAILURE')
            return self.result(None, 'OBSERVATION_DEPENDENCY_FAILURE')
        if current.players[0]['hitstop'] or current.players[0]['hp'] <= 0:
            if self.runner is not None:
                self.invalidate('HITSTOP_OR_TERMINAL')
            self.state = PlanState.WAITING
            return self.result(None, 'HITSTOP_OR_TERMINAL')
        if self.runner is None:
            if not trusted_boundary or legal_actions is None:
                self.state = PlanState.WAITING
                return self.result(None, 'WAIT_TRUSTED_BOUNDARY')
            if frame > 0 and (receipt is None or not receipt.consumed or receipt.poll_index is None):
                self.state = PlanState.WAITING
                return self.result(None, 'WAIT_ACTUAL_INPUT_HISTORY')
            if self.invalidated_at is not None:
                self.counts['recoveries'] += 1
                self.counts['recovery_frames'] += frame - self.invalidated_at
                self.invalidated_at = None
            self.clear_plan()
            # Start from real submitted keys, never the rejected proposal.
            actual = self.initial_action if receipt is None else receipt.action
            keys = decode_action(self.interface.command(actual)).inputs
            x, y, *buttons = keys
            raw = [int(y < 0), int(y > 0), int(x < 0), int(x > 0), *buttons]
            self.api.requested[:] = raw
            self.api.applied[:] = raw
            self.api.events.clear()
            self.last_poll = None if receipt is None else receipt.poll_index
            self.serial += 1
            self.runner = self.coroutine(self.env[b'teacher_decide'])
            self.counts['plans'] += 1
        if legal_actions is None:
            self.invalidate('MISSING_CURRENT_LEGAL_MASK')
            return self.result(None, 'MISSING_CURRENT_LEGAL_MASK')
        self.mode(b'decision')
        try:
            done, provenance = self.call(self.runner)
            candidate = self.interface.action(encode_action(self.api.inputs()))
            if candidate % 10 in (5, 6) or not legal_actions[candidate]:
                self.invalidate('MASKED_OR_UNREPRESENTABLE_INPUT')
                return self.result(None, 'MASKED_OR_UNREPRESENTABLE_INPUT')
        except (LuaError, ValueError, TypeError, KeyError) as error:
            self.invalidate('UNTRUSTED_DECISION_DEPENDENCY')
            return self.result(None, 'UNTRUSTED_DECISION_DEPENDENCY', detail=str(error))
        finally:
            self.mode(b'observation')
        self.state = PlanState.ACTIVE
        self.pending = dict(candidate=candidate, facing=facing, done=done)
        self.counts['labels'] += 1
        source = None if provenance is None else (
            provenance[b'source'].decode('ascii'), int(provenance[b'line']))
        helpers = [] if provenance is None else [v.decode('ascii') for _, v in provenance[b'helpers'].items()]
        family = next((n for n in helpers if n.startswith('lvr')), 'raw_keys_or_wait')
        self.counts['labels:'+family] += 1
        return self.result(candidate, 'TRUSTED_INPUT_CANDIDATE', source=source,
            intent=family)

    def result(self, label, reason, **extra):
        if label is None:
            self.mask_reasons[reason] += 1
        row = dict(frame=self.frame, plan_id=self.serial, state=self.state.value,
            reason=reason, training_label=label,
            reconciliation=self.last_reconciliation,
            execution='INTENT' if label is not None else 'NOT_SUBMITTED', **extra)
        self.trace.append(row)
        return row

    def report(self):
        return dict(counts=dict(self.counts), invalid_reasons=dict(self.mask_reasons),
            plan_invalidation_reasons=dict(self.reasons),
            valid_label_fraction=self.counts['labels']/max(1,self.counts['queries']),
            average_recovery_frames=self.counts['recovery_frames']/max(1,self.counts['recoveries']),
            state=self.state.value, native_verified=False, training_ready=False,
            dependency='caller-proven input consumption and legal masks required',
            semantics='source adaptation; observation-before-decision; main writes plan-local')

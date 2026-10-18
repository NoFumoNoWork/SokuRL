"""Selective, source-pinned passive memory over delayed student observations.

This is a restricted AI adaptation, not native timer/state reconstruction. No
original main/calculation thread runs, and no decision globals are rolled back.
Unknown prehistory stays unknown; local source groups can recover independently.
"""
from dataclasses import dataclass
from collections import deque

from lupa.lua51 import LuaError

from .source_adapter import SOURCES, digest


@dataclass(frozen=True)
class Rule:
    name: str
    first: int
    last: int
    writes: tuple[str, ...]
    kind: str


RULES = (
    Rule('air_ticks',706,710,('air_math',),'OBSERVATION_MEMORY'),
    Rule('ground_ticks',711,715,('graund_math',),'OBSERVATION_MEMORY'),
    Rule('after_guard',718,747,('after_guard_count',),'EVENT_MEMORY'),
    # Include the original block comments: their closing token shares the first
    # executable `if` line. Extracting only that line changes Lua syntax.
    Rule('my_allframe',810,833,('my_allframe',),'OBSERVATION_MEMORY'),
    Rule('enemy_allframe',834,857,('enemy_allframe',),'OBSERVATION_MEMORY'),
    Rule('my_front',935,939,('my_frontside',),'OBSERVATION_DERIVED'),
    Rule('enemy_front',941,945,('enemy_frontside',),'OBSERVATION_DERIVED'),
    Rule('my_back',947,951,('my_backside',),'OBSERVATION_DERIVED'),
    Rule('enemy_back',953,957,('enemy_backside',),'OBSERVATION_DERIVED'),
)
HISTORY_RULES = (Rule('my_history',797,802,('my_preact','my_preframe'),'OBSERVATION_MEMORY'),
    Rule('enemy_history',804,809,('enemy_preact','enemy_preframe'),'OBSERVATION_MEMORY'))
HISTORY_NAMES = ('my_preact','my_preframe','enemy_preact','enemy_preframe')
SUPPORTED = {name for rule in RULES+HISTORY_RULES for name in rule.writes}
EXECUTION = {'countD','loop_count'}
SOURCE_CONSTANTS = {'RnI_B','RnI_C','B_count','C_count'}
PARAMETERS = {'rapid','stand_neutral','stand_front','stand_back','caution_range',
              'base_chain','muteki','graze_atk','m300_rapid'}
SCRATCH = {'i','true_math','in_of_ranges','out_of_ranges','JFP_No','JFP_hp','true_obj_n'}
TACTICAL = {'ex_combo','crush_combo'}
HIDDEN_MEMORY = {'bio_math','my_airEX','enemy_airEX','my_xspeedEX','enemy_xspeedEX',
    'old_bt','old_x','old_ex','old_m_ys','old_e_ys','m_gs','e_gs','old_obj_dis',
    'obj_speedEX','obj_disEX','old_obj_disEX','pre0_enemy_obj_n','flag_kishitu',
    'recovary_time','atk_count','shoot_count','datk_count'}


def classify(name):
    """Role is not trust. Unreviewed variables remain UNKNOWN, never zero-filled."""
    if name in SUPPORTED:
        return (next(r.kind for r in RULES+HISTORY_RULES if name in r.writes),
                'SELECTIVE_SOURCE_RULE_RUNTIME_VALIDITY_REQUIRED')
    if name in EXECUTION:
        return 'EXECUTION_PROGRESS','UNTRUSTED_REQUIRES_ACTUAL_INPUT_AND_ACCEPTANCE'
    if name in SOURCE_CONSTANTS:
        return 'SCRIPT_CONSTANT','PINNED_PACKAGE_UNUSED_HOLD_FLAGS_NOT_STUDENT_KEY_STATE'
    if name in PARAMETERS:
        return 'STATIC_AI_PARAMETER','SOURCE_PARAMETER_NOT_GAMEPLAY_MEASUREMENT'
    if name in SCRATCH:
        return 'QUERY_SCRATCH','DISPOSABLE_NOT_PERSISTENT_TACTICAL_MEMORY'
    if name in TACTICAL:
        return 'TACTICAL_INTENT','UNTRUSTED_DECISION_WRITES_NOT_ENABLED'
    if name in HIDDEN_MEMORY or name.startswith(('enemy_obj_','pre0_enemy_obj_', 'enemy_hitarea_')):
        return 'OBSERVATION_OR_EVENT_MEMORY','UNTRUSTED_MISSING_SHARED_DEPENDENCIES'
    return 'UNKNOWN','UNTRUSTED_REQUIRES_SOURCE_REVIEW'


COMPILE = b'''function(code,name)
 local f,message=loadstring(code,name)
 if not f then error(message) end
 return function(env) setfenv(f,env); return f() end
end'''
READONLY = b'''function(data)
 return setmetatable({}, {__index=function(_,key)
   local v=rawget(data,key)
   if v==nil then error('UNTRUSTED_TEACHER_MEMORY:'..tostring(key)) end
   return v
 end, __newindex=function() error('READ_ONLY_TEACHER_MEMORY') end})
end'''
ENVIRONMENT = b'''function(inputs,outputs,allowed)
 return setmetatable({}, {__index=function(_,key)
   local v=rawget(outputs,key)
   if v~=nil then return v end
   v=rawget(inputs,key)
   if v==nil then error('UNTRUSTED_TEACHER_MEMORY:'..tostring(key)) end
   return v
 end, __newindex=function(_,key,value)
   if not allowed[key] then error('UNDECLARED_MEMORY_WRITE:'..tostring(key)) end
   outputs[key]=value
 end})
end'''


class ObservedMemory:
    def __init__(self, episode):
        self.episode=episode
        data=episode.teacher.package.files['def_clc.lua']
        if digest(data)!=SOURCES['def_clc.lua']:
            raise ValueError('Unreviewed passive-memory source')
        self.source=data
        self.lines=data.splitlines(keepends=True)
        self.compile=episode.lua.eval(COMPILE)
        self.readonly=episode.lua.eval(READONLY)
        self.environment=episode.lua.eval(ENVIRONMENT)
        self.bind=episode.lua.eval(b'function(f,e) return function() return f(e) end end')
        self.runners={r.name:self.compile(self.slice(r.first,r.last),
            ('@community/def_clc.lua#'+r.name).encode()) for r in RULES+HISTORY_RULES}
        self.finish=self.compile(self.slice(1094,1097),b'@community/def_clc.lua#history_commit')
        self.values={}
        self.histories={}
        self.status={}
        self.last_index=None
        self.observations=0
        self.gaps=0
        self.invalidated=deque(maxlen=64)
        self.failure=None

    def slice(self, first, last):
        return b'\n'*(first-1)+b''.join(self.lines[first-1:last])

    def execute(self, runner, env):
        episode=self.episode
        episode.budget(episode.teacher.spec['instruction_limit'])
        try:
            return episode.intents.invoke(self.bind(runner,env),True)
        except LuaError as error:
            if 'UNTRUSTED_TEACHER_MEMORY:' not in str(error):
                self.failure=str(error)
            raise

    def interrupt(self, reason):
        """An observation gap invalidates continuity, not the whole episode VM."""
        self.invalidated.append(dict(after=self.last_index,reason=reason))
        self.values.clear()
        self.histories.clear()
        self.status.clear()
        self.last_index=None
        for name in SUPPORTED:
            self.episode.env[name.encode()]=None
        if self.episode.intents.pending is not None:
            self.episode.intents.reject(self.episode.intents.pending.intent_id)

    def observe(self, index):
        if self.failure is not None:
            raise RuntimeError('BLOCKED_PASSIVE_MEMORY_FAILURE: reset required; '+self.failure)
        if type(index) is not int or index<0:
            raise ValueError('passive memory requires a nonnegative delayed index')
        if self.last_index is not None and index<=self.last_index:
            raise ValueError('passive memory cannot repeat or rewind an observation')
        if self.last_index is not None and index!=self.last_index+1:
            self.gaps+=1
            self.interrupt('NONCONSECUTIVE_DELAYED_OBSERVATION')
        episode=self.episode
        # Explicit source dependencies only. Never import the initialized original
        # calculation globals, hidden APIs or teacher-only input state.
        names=('my_act','enemy_act','my_frame','enemy_frame','my_char','enemy_char',
               'my_dir','enemy_dir','x','ex','ey','stage_number','bgm_number')
        fields={name.encode():episode.env[name.encode()] for name in names}
        if not self.histories:
            for prefix in ('my','enemy'):
                self.histories[prefix+'_preact']=episode.lua.table_from({0:fields[(prefix+'_act').encode()]})
                self.histories[prefix+'_preframe']=episode.lua.table_from({0:fields[(prefix+'_frame').encode()]})
        raw=episode.lua.table_from(fields | {k.encode():v for k,v in self.histories.items()})
        for rule in HISTORY_RULES:
            # Sparse table copies intentionally preserve unknown historical slots.
            # Source-owned loop variable i lives only in this observer namespace.
            self.execute(self.runners[rule.name],raw)
        bindings=fields | {k.encode():v for k,v in self.values.items()}
        bindings.update({k.encode():self.readonly(v) for k,v in self.histories.items()})
        for rule in RULES:
            outputs=episode.lua.table()
            allowed=episode.lua.table_from({k.encode():True for k in rule.writes})
            env=self.environment(episode.lua.table_from(bindings),outputs,allowed)
            try:
                self.execute(self.runners[rule.name],env)
                values={k:outputs[k.encode()] for k in rule.writes}
                if any(v is None for v in values.values()):
                    raise ValueError('Source memory rule produced no declared value')
            except LuaError as error:
                # A missing read taints only this memory group; budget exhaustion
                # and unexpected implementation errors are NOT hidden as missing data.
                if 'UNTRUSTED_TEACHER_MEMORY:' not in str(error):
                    raise
                for name in rule.writes:
                    self.values.pop(name,None)
                    bindings.pop(name.encode(),None)
                self.status[rule.name]=dict(status='UNTRUSTED_DEPENDENCY',detail=str(error))
            else:
                self.values.update(values)
                bindings.update({k.encode():v for k,v in values.items()})
                self.status[rule.name]=dict(status='TRUSTED_PUBLIC_HISTORY',values=values)
        # Both allframe calculations read previous observations before the source
        # commits current index0. This order is material for repeated action IDs.
        self.execute(self.finish,raw)
        for name in HISTORY_NAMES:
            episode.env[name.encode()]=self.readonly(self.histories[name])
        for rule in RULES:
            for name in rule.writes:
                episode.env[name.encode()]=self.values.get(name)
        self.last_index=index
        self.observations+=1
        return self.report()

    def report(self):
        histories={name:{str(i):table[i] for i in range(6) if table[i] is not None}
                   for name,table in self.histories.items()}
        return dict(schema='teacher_observed_memory_v1',
            evidence='PINNED_AI_HEURISTIC_OVER_PUBLIC_DELAYED_HISTORY_NOT_NATIVE_STATE',
            source_sha256=SOURCES['def_clc.lua'],observation_index=self.last_index,
            observations=self.observations,gaps=self.gaps,invalidations=list(self.invalidated),
            failure=self.failure,
            values=dict(self.values),histories=histories,groups=dict(self.status),
            source_rules=[dict(name=r.name,lines=[r.first,r.last],writes=list(r.writes),
                sha256=digest(self.slice(r.first,r.last))) for r in RULES+HISTORY_RULES],
            semantics=dict(clock='one update per actual consecutive delayed student query, not native battle_time',
                history='unknown prehistory is absent, never source-initialized zero',
                after_guard='original AI response heuristic, NOT native guard/hitstun timer',
                wall_distance='original AI constants40/1240, NOT proof of actual stage boundaries',
                tactical='original decision writes not executed; no global rollback',
                execution='no coroutine/key prefix advanced or assumed accepted'),
            labels_enabled=False,native_verified=False)

"""Original source decisions with explicit state ownership, not global rollback.

Only observation-confirmed tactical flags and static AI parameters persist.
Decision scratch/plan writes live in a disposable scope, never an assumed input
continuation. Missing dependencies mask this query without poisoning the episode.
"""
from types import SimpleNamespace
import json
from pathlib import Path

from lupa.lua51 import LuaError, LuaMemoryError

from .intents import BRIDGE
from .observed_memory import SUPPORTED, READONLY, PARAMETERS, SCRATCH, SOURCE_CONSTANTS
from .source_adapter import SOURCES, digest, split_loop
from .guard_order import load_registry, replacements


API_SHA256='61cb6a24af1d60d86305481fd8bf4a6b12cc801a57c4a23f91be0e8355489d58'
PLAN={'ex_combo','crush_combo','loop_count','flag_kishitu'}
PUBLIC=SUPPORTED | {'my_char','enemy_char','my_act','enemy_act','my_dir','enemy_dir',
    'my_act_block','enemy_act_block','my_hp','enemy_hp','my_frame','enemy_frame',
    'my_subsequence','enemy_subsequence','my_pose_frame','enemy_pose_frame',
    'x','y','ex','ey','my_x','my_y','enemy_x','enemy_y','dis','dis_x','dis_y','dis2',
    'front','back','d_front','d_back','u_front','u_back','is_dir_front',
    'stage_number','bgm_number','get_visible_obj_data',
    'visible_my_obj_n','visible_enemy_obj_n','visible_selection_complete',
    'visible_selection_known','visible_selection_status','task_fixed_weather',
    'task_cards_disabled','teacher_query_index','teacher_observation_index',
    'teacher_episode_progress','my_spirit_fraction','enemy_spirit_fraction',
    'my_screen_x','my_screen_y','enemy_screen_x','enemy_screen_y'}
BUILTINS={'assert','error','ipairs','next','pairs','select','tonumber','tostring',
          'type','unpack','print','math','table','string','coroutine'}


SCOPE=br'''function(public,allowed,parameters,scratch,plans,constants,context)
 local raw,info,traceback=rawget,debug.getinfo,debug.traceback
 local defs,initial,static,localvars,observed={},{},{},{},{}
 local libraries={}
 for _,name in ipairs({'math','table','string','coroutine'}) do
   local values=raw(public,name)
   libraries[name]=setmetatable({}, {__index=values,
     __newindex=function() error('READ_ONLY_TEACHER_LIBRARY') end})
 end
 local loading,setting=true,false
 local env
 local function missing(key)
   local at=info(3,'Sl')
   error('UNTRUSTED_TEACHER_STATE:'..tostring(key)..':'..at.source..':'..at.currentline..'\n'..traceback('',3))
 end
 env=setmetatable({}, {__index=function(_,key)
   if key=='_G' then return env end
   if libraries[key] then return libraries[key] end
   if defs[key]~=nil then return defs[key] end
   if context[key]~=nil then return context[key] end
   if loading and initial[key]~=nil then return initial[key] end
   if constants[key] then return 0 end
   if setting and static[key]~=nil then return static[key] end
   if static[key]~=nil then return static[key] end
   if localvars[key]~=nil then return localvars[key] end
   if observed[key]~=nil then return observed[key] end
   if allowed[key] then
     local v=raw(public,key)
     if v~=nil then return v end
   end
   return missing(key)
 end, __newindex=function(_,key,value)
   if loading then
     if type(value)=='function' then defs[key]=value else initial[key]=value end
   elseif setting and parameters[key] then static[key]=value
   elseif scratch[key] or plans[key] then localvars[key]=value
   else error('UNREVIEWED_TEACHER_WRITE:'..tostring(key)) end
 end})
 return env, function(value) loading=value end,
   function(value) setting=value; if value then static={} end end,
   function(values) localvars={};observed=values or {} end,
   function(name) return defs[name] end,
   function() return localvars,static end,
   function() local n=0;for _ in pairs(initial) do n=n+1 end;initial={};return n end
end'''


class ScopedDecisions:
    def __init__(self, episode):
        self.episode=episode
        if digest(episode.teacher.package.api)!=API_SHA256:
            raise ValueError('Scoped teacher requires the reviewed original API source')
        self.readonly=episode.lua.eval(READONLY)
        allowed=PUBLIC | BUILTINS
        # API scalar constants are definitions, not additional native reads.
        self.constants={k:v for k,v in episode.env.items()
                        if isinstance(k,bytes) and k.startswith(b'ACT_') and type(v) in (int,float)}
        allowed.update(k.decode() for k in self.constants)
        self.context={}
        if episode.teacher.context is not None:
            from sokurl_platform.bridge_shared import WEATHER_CLEAR
            # A declared fixed-task premise, not a native observation. Existing
            # enforceFixedWeather sets these three fields; Phase4 must validate
            # the actual deployment. No hand/card/skill values are fabricated.
            self.context=dict(weather=WEATHER_CLEAR,weather2=WEATHER_CLEAR,weather_time=0)
        bits=lambda names:episode.lua.table_from({k.encode():True for k in names})
        (self.env,self.loading,self.setting,self.begin,self.entry,self.state,
         self.clear_initial)=episode.lua.eval(SCOPE)(episode.env,bits(allowed),
            bits(PARAMETERS),bits(SCRATCH),bits(PLAN),bits(SOURCE_CONSTANTS),
            episode.lua.table_from({k.encode():v for k,v in self.context.items()}))
        self.compile=episode.lua.eval(b'''function(code,name,env)
          local f,e=loadstring(code,name);if not f then error(e) end
          setfenv(f,env);return f
        end''')
        self.loaded=False
        self.enemy=None
        self.quarantined=0
        self.tactical={}
        self.tactical_status={}
        self.last_observation=None
        self.last_plan={}
        self.queries=0
        self.masked=0
        self.fatal=None
        self.rules={}
        self.guard_adaptation=None
        self.generated={}
        self.task_gates=[]
        self.flag_guard_adaptations=[]
        self.pure_guards,self.pure_guards_sha256=load_registry()
        if self.pure_guards.get('api_sha256')!=API_SHA256:
            raise ValueError('Unreviewed pure-guard API source')
        self.pure_guard_adaptations=[]
        # Reuse the existing public layout's applicability metadata ONLY, never
        # its addresses/readers or any native value. Original Character.cpp's
        # mismatching-character branches return -1 without a memory read.
        self.special_applicability=dict(episode.teacher.special_applicability)
        registry_bytes=Path(__file__).with_name('card_only_gates.json').read_bytes()
        self.registry_sha256=digest(registry_bytes)
        self.registry=json.loads(registry_bytes)
        if (self.registry.get('schema')!='teacher_card_only_gates_v1' or
                self.registry.get('sources')!=SOURCES or self.registry.get('api_sha256')!=API_SHA256):
            raise ValueError('Unreviewed card-only gate registry identity')
        if episode.teacher.spec['script']=='00_reimu_main.ai':
            source=episode.teacher.package.files['00_reimu_main.ai'].splitlines(keepends=True)
            for name,first,last,key in (('ex_reset',53,55,'ex_combo'),
                ('crush_reset',57,59,'crush_combo'),('ex_spell_observed',61,63,'ex_combo')):
                code=b'\n'*(first-1)+b''.join(source[first-1:last])
                self.rules[name]=(episode.observed_memory.compile(code,b'@community/00_reimu_main.ai#'+name.encode()),key,first,last,digest(code))

    def run(self, function):
        self.episode.budget(self.episode.teacher.spec['instruction_limit'])
        return self.episode.intents.invoke(function,True)

    def inapplicable_special(self, player, index):
        if type(player) is int and player in (0,1) and type(index) is int:
            required=self.special_applicability.get(index)
            character=self.episode.env[b'my_char' if player==0 else b'enemy_char']
            if required is not None and character!=required:
                return -1  # known API inapplicability, NOT an invented timer
        raise ValueError('UNTRUSTED_TEACHER_STATE:get_special_data:applicable_or_unreviewed_field')

    def require(self, name):
        if isinstance(name,bytes):name=name.decode('ascii')
        if name not in SOURCES:
            raise ValueError('Scoped teacher dependency outside pinned package')
        data=self.episode.teacher.package.files[name]
        if digest(data)!=SOURCES[name]:raise ValueError('Scoped teacher source hash mismatch')
        code=self.episode.teacher.adapter.sources.get(name,data)
        if name==self.episode.teacher.spec['script']:
            edits={(first,last):value for first,last,value in replacements(data,name,self.pure_guards)}
            gates=([g for g in self.registry['gates'] if g['source']==name]
                   if self.episode.teacher.context is not None else [])
            previous=len(data)
            for gate in sorted(gates,key=lambda g:g['condition_start'],reverse=True):
                first,last=gate['condition_start'],gate['condition_end']
                if (type(first) is not int or type(last) is not int or not 0<=first<last<=previous
                        or gate['kind'] not in ('If','ElseIf') or gate['body_effects']!=['CARD']
                        or not set(gate['guard_effects'])<={'CARD'}
                        or digest(data[first:last])!=gate['condition_sha256']):
                    raise ValueError('Invalid card-only source condition binding')
                original=edits.get((first,last),data[first:last])
                edits[first,last]=b'(not task_cards_disabled) and ('+original+b')'
                previous=first
            previous=len(data)
            for (first,last),value in sorted(edits.items(),reverse=True):
                if last>previous:raise ValueError('Overlapping reviewed guard edits')
                data=data[:first]+value+data[last:]
                previous=first
            code,_=split_loop(data,b'main',b'teacher_decide')
            self.task_gates=gates
            self.pure_guard_adaptations=[r for r in self.pure_guards['guards'] if r['source']==name]
        if name=='def_clc.lua':
            # Do not execute thousands of untrusted object/history initializer
            # writes merely to load functions. These pinned complete helper
            # regions contain all non-calculation functions, with original line
            # numbers. Only reviewed static defaults are executed separately.
            code=b''.join(line if 163<=i<=403 or 418<=i<=639 else b'\n'
                          for i,line in enumerate(code.splitlines(keepends=True),1))
        if name==self.episode.teacher.spec['script']:
            before=b'y-my_yspeed*3>0 and my_act>=50 and my_act<=107 and'
            after=b'my_act>=50 and my_act<=107 and y-my_yspeed*3>0 and'
            if code.count(before)!=1:raise ValueError('Pinned pure velocity-guard seam changed')
            code=code.replace(before,after,1)
            self.guard_adaptation=dict(line=104 if name.startswith('01_') else 116,
                original=before.decode(),adapted=after.decode(),guard_only_sha256=digest(code),
                evidence='commuting numeric global reads only; no calls/RNG/writes; truth value unchanged')
            self.flag_guard_adaptations=[]
            for before,after in (
                (b'get_fflags(0,2097152)==true and y==0 and my_act>=300',
                 b'my_act>=300 and y==0 and get_fflags(0,2097152)==true'),
                (b'get_fflags(0,2097152)==true and y>0 and my_airEX>=1 and my_act>=300',
                 b'my_act>=300 and y>0 and get_fflags(0,2097152)==true and my_airEX>=1')):
                if code.count(before)!=1:raise ValueError('Pinned pure HJC-guard seam changed')
                code=code.replace(before,after,1)
                self.flag_guard_adaptations.append(dict(original=before.decode(),adapted=after.decode(),
                    evidence='get_fflags is a read-only bit test: th123_ai/scriptLua.cpp:506; no substituted flag value'))
        self.generated[name]=digest(code)
        return self.run(self.compile(code,('@community/'+name).encode(),self.env))

    def prepare(self):
        expected=1 if self.episode.teacher.spec['script']=='01_marisa_main.ai' else 0
        if self.episode.env[b'my_char']!=expected:
            raise ValueError('Scoped script/player character mismatch')
        if not self.loaded:
            self.env[b'require']=self.require
            self.env[b'get_special_data']=self.episode.lua.eval(b'''function(callback)
              return function(player,index) return callback(player,index) end
            end''')(self.inapplicable_special)
            self.require(self.episode.teacher.spec['script'])
            install,_,_=self.episode.lua.eval(BRIDGE)(self.env)
            install()
            self.episode.teacher.adapter.install(SimpleNamespace(lua=self.episode.lua,env=self.env))
            self.quarantined=self.clear_initial()
            self.loading(False)
            self.loaded=True
        enemy=self.episode.env[b'enemy_char']
        if enemy!=self.enemy:
            self.setting(True)
            try:
                self.run(self.compile(b'\n'*404+b''.join(self.episode.teacher.package.files['def_clc.lua'].splitlines(keepends=True)[404:416]),
                    b'@community/def_clc.lua#static_defaults',self.env))
                self.run(self.entry(b'base_states'))
                _,static=self.state()
                static[b'rapid']=self.readonly(static[b'rapid'])
            finally:
                self.setting(False)
            self.enemy=enemy

    def observe(self, index):
        if self.last_observation is not None and index!=self.last_observation+1:
            self.tactical.clear()
            self.discard_plan('OBSERVATION_GAP')
        self.last_observation=index
        memory=self.episode.observed_memory
        for name,(function,key,first,last,sha) in self.rules.items():
            inputs={k.encode():v for k,v in self.tactical.items()}
            for field in ('my_act','enemy_act'):
                inputs[field.encode()]=self.episode.env[field.encode()]
            outputs=self.episode.lua.table()
            env=memory.environment(self.episode.lua.table_from(inputs),outputs,
                self.episode.lua.table_from({key.encode():True}))
            try:
                memory.execute(function,env)
            except LuaError as error:
                if 'UNTRUSTED_TEACHER_MEMORY:' not in str(error):raise
                self.tactical.pop(key,None)
                self.tactical_status[name]='UNTRUSTED_DEPENDENCY'
            else:
                value=outputs[key.encode()]
                if value is not None:self.tactical[key]=value
                self.tactical_status[name]='OBSERVED_SOURCE_UPDATE' if value is not None else 'NO_WRITE_CURRENT_STATE'

    def invalidate_observation(self):
        self.tactical.clear()
        self.tactical_status.clear()
        self.last_observation=None
        self.discard_plan('UNAVAILABLE_OBSERVATION')

    def discard_plan(self, reason):
        self.last_plan={}
        self.begin()

    def query(self, opportunity, boundary_id):
        from .query import Advice
        session=self.episode.intents
        blocked=session.boundary(opportunity,boundary_id)
        if blocked is not None:return blocked
        if self.fatal is not None:return self.fatal
        self.discard_plan('NEW_REAL_BOUNDARY')
        self.queries+=1
        try:
            self.prepare()
            # Only source-confirmed flags enter the decision, never prior plan
            # assignments. They live in this query's local ownership scope.
            self.begin(self.episode.lua.table_from({k.encode():v for k,v in self.tactical.items()}))
            self.episode.budget(self.episode.teacher.spec['instruction_limit'])
            value=session.invoke(self.entry(b'teacher_decide'))
            localvars,_=self.state()
            self.last_plan={key:localvars[key.encode()] for key in PLAN if localvars[key.encode()] is not None}
            answer=session.interpret(value,opportunity,boundary_id)
            if answer.status!='INTENT_PROPOSED':self.discard_plan('NO_TRUSTED_PROPOSAL')
            return answer
        except (LuaError,LuaMemoryError,ValueError,TypeError,AttributeError) as error:
            self.discard_plan('QUERY_FAILURE')
            text=str(error)
            self.masked+=1
            answer=Advice('MASKED_SOURCE_STATE',source=self.episode.location().decode('utf-8','replace'),
                          detail=text,decision_frame=boundary_id)
            if not isinstance(error,LuaMemoryError) and any(tag in text for tag in (
                'UNTRUSTED_TEACHER_STATE:','UNTRUSTED_TEACHER_MEMORY:','UNREVIEWED_TEACHER_WRITE:',
                'READ_ONLY_TEACHER_LIBRARY','READ_ONLY_TEACHER_MEMORY')):
                return answer
            self.fatal=Advice('BLOCKED_SCOPED_SOURCE',source=answer.source,detail=text)
            return self.fatal

    def report(self):
        _,static=self.state()
        parameters={key:static[key.encode()] for key in PARAMETERS-{'rapid'} if static[key.encode()] is not None}
        return dict(schema='teacher_scoped_state_v1',loaded=self.loaded,queries=self.queries,
            masked=self.masked,quarantined_initial_globals=self.quarantined,
            tactical=dict(self.tactical),tactical_status=dict(self.tactical_status),
            uncommitted_plan=dict(self.last_plan),fatal=self.fatal.record() if self.fatal else None,
            sources=SOURCES,api_sha256=API_SHA256,observation_index=self.last_observation,
            generated_sha256=dict(self.generated),
            static_parameters=parameters,
            task_premises=dict(values=self.context,basis='DECLARED_FIXED_TASK_NOT_NATIVE_READ',
                source='Platform bridge_shared.WEATHER_CLEAR/enforceFixedWeather; th123_ai main.cpp weather export',
                native_verified=False),
            pure_guard_adaptation=self.guard_adaptation,
            card_only_task_gates=list(self.task_gates),
            card_gate_registry_sha256=self.registry_sha256,
            pure_flag_guard_adaptations=list(self.flag_guard_adaptations),
            pure_guard_order=dict(registry_sha256=self.pure_guards_sha256,
                guards=self.pure_guard_adaptations,
                semantics='known false necessary conjuncts avoid irrelevant missing skill reads; no unavailable getter supplied'),
            special_applicability=dict(required_characters=self.special_applicability,
                source='existing sokurl_platform.native_layout.SPECIAL_FIELDS; th123_ai Character::GetSpecialData mismatch branches',
                semantics='-1 only when public character proves API inapplicable; applicable/common/unreviewed fields remain unavailable',
                native_values_exposed=False),
            definition_regions=dict(source='def_clc.lua',lines=[[163,403],[418,639]],
                omitted='original history/object initialization and complete-list calculation; not restored as trusted zeros'),
            rules=[dict(name=name,source='00_reimu_main.ai',lines=[row[2],row[3]],sha256=row[4])
                   for name,row in self.rules.items()],
            semantics='persistent definitions/static parameters/observed flags; query-local scratch and plan; no global rollback',
            primitive_labels=False,native_verified=False)

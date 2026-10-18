"""The teacher reads ONLY the student's versioned common tensor, not a process."""
from lupa.lua51 import LuaRuntime

from soku_rl.env.observation.shared_native import decode_shared, SCHEMA
from .api import ScriptAPI
from .runtime import GodActor


class SharedNativeAPI(ScriptAPI):
    def get_key_stat2(self,player,key):
        if player!=0: raise ValueError('SHARED_PRIVATE_INPUT_UNAVAILABLE: opponent keys')
        return super().get_key_stat2(player,key)

    def get_key_map2(self,player):
        if player!=0: raise ValueError('SHARED_PRIVATE_INPUT_UNAVAILABLE: opponent keys')
        return super().get_key_map2(player)

    def get_deck_list(self,*args):
        raise ValueError('SHARED_FUTURE_DECK_ORDER_UNAVAILABLE')


class SharedTeacherPerception:
    """Perception/queries only. No source coroutine, key injection or PPO labels."""
    def __init__(self,seed):
        self.lua=LuaRuntime(encoding=None,unpack_returned_tuples=True,
            register_eval=False,register_builtins=False,
            attribute_filter=lambda *args:(_ for _ in ()).throw(AttributeError('Python access denied')))
        self.api=SharedNativeAPI(self.lua,seed)
        self.frame=-1
        self.queries=0

    def observe(self,student_values,*,schema):
        current=decode_shared(student_values,schema=schema)
        frame=current.world['frame']
        if frame!=self.frame+1:
            raise ValueError('shared teacher requires consecutive frames; Reset creates a new perception')
        self.api.observe(current)
        self.frame=int(frame)
        self.queries+=1
        # Return the same sanitized values that fed ScriptAPI, for independent
        # agreement checks. Never fetch a richer teacher-only observation.
        return current

    def report(self):
        return dict(schema=SCHEMA,frames=self.queries,frame=self.frame,
            additional_reaction_frames=0,source='same student tensor',
            decisions='NOT_ATTACHED',control='NONE',training_ready=False)


class SharedNativeSourceActor(GodActor):
    """Reuse the existing full source runner for CONTROLLED diagnostics only.

    It may advance only if its preceding proposal was actually executed. This
    deliberately exposes the unresolved ownership issue instead of assuming a
    rejected coroutine prefix happened, or silently turning it into PPO labels.
    It is NOT the completed continuous off-policy replanning teacher.
    """
    def __init__(self,policy,seed):
        from .source_adapter import SOURCES,digest
        if policy.script not in ('01_marisa_main.ai','00_reimu_main.ai'):
            raise ValueError('shared v1 source diagnostic supports Reimu/Marisa only')
        for name,sha in SOURCES.items():
            if digest(policy.package.files[name])!=sha:
                raise ValueError('shared native source actor requires the reviewed2011 package')
        if digest(policy.package.api)!='61cb6a24af1d60d86305481fd8bf4a6b12cc801a57c4a23f91be0e8355489d58':
            raise ValueError('shared native source actor requires the reviewed original API')
        super().__init__(policy,seed,api_type=SharedNativeAPI)
        self.previous_proposal=None
        self.control_blocked=False

    def propose(self,student_values,*,schema,previous_executed=None):
        current=decode_shared(student_values,schema=schema)
        if current.players[0]['char']!=(1 if self.policy.script.startswith('01_') else 0):
            raise ValueError('shared teacher script/player mismatch')
        if self.control_blocked:
            return dict(status='BLOCKED_CONTROL_ALIGNMENT',training_label=None)
        if self.previous_proposal is not None and previous_executed!=self.previous_proposal:
            self.control_blocked=True
            return dict(status='BLOCKED_CONTROL_ALIGNMENT',training_label=None,
                expected_previous=self.previous_proposal,actual_previous=previous_executed,
                detail='Original coroutine not resumed. Local intent replanning remains a separate requirement.')
        command=self.act_observation(current)
        self.previous_proposal=command
        return dict(status='CONTROLLED_SOURCE_PROPOSAL',command=command,training_label=None,
            native_acceptance='NOT_OBSERVED',
            note='No game control or PPO label. Prefix equality is not native move acceptance.')

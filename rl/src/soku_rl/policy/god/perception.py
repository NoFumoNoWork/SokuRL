"""Teacher view of the unchanged student animation tensor, not native state.

Selected object counts never stand for the complete native object list. Times,
resource fractions and animation counters retain their explicit source meaning.
"""
from dataclasses import asdict, dataclass

import numpy as np

from soku_rl.env.observation.animation import frame_width


FIELD_MAP = {
    'my_frame/enemy_frame': 'fighter token[9]: native frameCount already in student tensor',
    'my_subsequence/enemy_subsequence': 'fighter token[7]: animationCounter',
    'my_pose_frame/enemy_pose_frame': 'fighter token[8]: frameData.number, NOT sprite image ID',
    'my_spirit_fraction/enemy_spirit_fraction': 'quantized currentSpirit/maxSpirit, NOT absolute spirit',
    'my_screen_x/y,enemy_screen_x/y': 'existing quantized screen coordinates in pixels',
    'visible_my_obj_n/visible_enemy_obj_n': 'counts in selected public slots, NOT native obj_n',
    'visible_selection_complete/known/status': 'proven-complete bool plus explicit unknown-at-capacity status; NEVER native-list completeness',
    'teacher_query_index/teacher_observation_index': 'consecutive query clock and delayed index, NOT native battle_time',
    'teacher_episode_progress': 'existing normalized frameId/horizon, not an exact battle timer',
    'task_fixed_weather/task_cards_disabled': 'declared configuration context, not proof of runtime enforcement',
}

ALIASES = {'my_frame','enemy_frame','my_subsequence','enemy_subsequence',
    'my_pose_frame','enemy_pose_frame','my_spirit_fraction','enemy_spirit_fraction',
    'my_screen_x','my_screen_y','enemy_screen_x','enemy_screen_y',
    'visible_my_obj_n','visible_enemy_obj_n','visible_selection_complete','visible_selection_known','visible_selection_status',
    'teacher_query_index','teacher_observation_index','teacher_episode_progress'}


@dataclass(frozen=True)
class VisibleObject:
    slot: int
    owner: int
    x: float
    y: float
    screen_x: float
    screen_y: float
    facing: int
    character: int
    action: int
    sequence: int
    animation_counter: int
    animation_subframe: int


@dataclass(frozen=True)
class VisibleView:
    capacity: int
    objects: tuple[VisibleObject, ...]

    @property
    def selection_complete(self):
        return True if len(self.objects) < self.capacity else None

    def report(self):
        return dict(capacity=self.capacity, selected=len(self.objects),
            own=sum(x.owner == 0 for x in self.objects),
            enemy=sum(x.owner == 1 for x in self.objects),
            selection_complete=self.selection_complete,
            truncation='POSSIBLE_AT_CAPACITY' if self.selection_complete is None else 'NO_VISIBLE_SLOT_TRUNCATION',
            native_list_complete='UNKNOWN_HIDDEN_OBJECTS_FILTERED',
            identity='slot is not a stable cross-frame object identity',
            objects=[asdict(x) for x in self.objects])


def project(current, slots, world_coordinates, query_index, reaction_frames):
    """Only the delayed student row enters this projection."""
    if current.shape != (frame_width(slots, world_coordinates),) or not np.isfinite(current).all():
        raise ValueError('teacher perception requires one finite student row')
    core = current[:frame_width(slots)]
    values = dict(teacher_query_index=query_index,
        teacher_observation_index=query_index-reaction_frames,
        teacher_episode_progress=float(core[-1]))
    for offset, prefix in ((0,'my_'),(10,'enemy_')):
        token = core[offset:offset+10]
        if token[0] != 1:
            raise ValueError('invisible fighter fields must not be exposed')
        for suffix, value in (('frame',token[9]),('subsequence',token[7]),('pose_frame',token[8])):
            if value < 0 or value != int(value):
                raise ValueError('non-integral animation counter')
            values[prefix+suffix] = int(value)
        values[prefix+'screen_x'],values[prefix+'screen_y'] = float(token[1]*640),float(token[2]*480)
        values[prefix+'spirit_fraction'] = float(core[-3+(offset//10)])
    objects = []
    for slot, token in enumerate(core[20:-5].reshape(slots,10)):
        if token[0] == 0:
            continue
        if token[0] != 1 or token[9] not in (0,1):
            raise ValueError('invalid public object mask/owner')
        if any(v != int(v) for v in token[3:]) or token[3] not in (-1,1):
            raise ValueError('invalid object category')
        # World coordinates are optional for old contracts, not invented from
        # screen coordinates/camera assumptions. New P0 uses the world schema.
        if not world_coordinates:
            continue
        x,y = current[frame_width(slots)+4+2*slot:frame_width(slots)+6+2*slot]
        objects.append(VisibleObject(slot,int(token[9]),float(x),float(y),
            float(token[1]*640),float(token[2]*480),*map(int,token[3:9])))
    view = VisibleView(slots,tuple(objects))
    if world_coordinates:
        values.update(visible_my_obj_n=sum(x.owner==0 for x in objects),
            visible_enemy_obj_n=sum(x.owner==1 for x in objects))
        # Do not expose an "UNKNOWN" string as the predicate: all strings are
        # truthy in Lua. False here means no completeness proof, not proven loss.
        values['visible_selection_complete'] = view.selection_complete is True
        values['visible_selection_known'] = view.selection_complete is not None
        values['visible_selection_status'] = b'COMPLETE_VISIBLE_SELECTION' if view.selection_complete else b'UNKNOWN_AT_CAPACITY'
    return values,view


def install_visible_api(episode, view):
    def visible_object(owner, index):
        if type(owner) is not int or owner not in (0,1) or type(index) is not int:
            raise ValueError('visible object query requires owner/index integers')
        selected = [x for x in view.objects if x.owner == owner]
        if not 0 <= index < len(selected):
            raise ValueError('visible object index outside selected view')
        fields = {k.encode():v for k,v in asdict(selected[index]).items()}
        return episode.lua.eval(b'''function(data)
          return setmetatable({}, {__index=function(_,key)
            local v=rawget(data,key)
            if v==nil then error('UNAVAILABLE_VISIBLE_OBJECT_FIELD:'..tostring(key)) end
            return v
          end, __newindex=function() error('READ_ONLY_VISIBLE_OBJECT') end})
        end''')(episode.lua.table_from(fields))
    episode.env[b'get_visible_obj_data'] = visible_object
    episode.visible_view = view


def task_context(runtime):
    """Bind known task declarations; do not manufacture raw weather/card reads."""
    if runtime.get('fixed_weather') != 'clear' or runtime.get('cards_disabled') is not True:
        raise ValueError('restricted teacher task requires explicit Clear/cards-disabled runtime configuration')
    return dict(fixed_weather='clear',cards_disabled=True,
                binding='CONFIGURATION_ONLY_NATIVE_ENFORCEMENT_NOT_VERIFIED_HERE')

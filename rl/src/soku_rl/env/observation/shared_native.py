"""Versioned SAME current-state inputs; no opponent keys or future deck order.

Consumes already decoded Platform data. This module never reads a process or
changes capture timing. Zero redaction slots are padding, not observed key/deck
values; every teacher access to these channels must be denied explicitly.
"""
from dataclasses import dataclass

import numpy as np

from .memory_schema import FIGHTER_NAMES, OBJECT_NAMES, PRIVILEGED_FEATURES, WORLD_NAMES
from .privileged import PrivilegedObservation, encode_privileged, decode_privileged
from .animation import frame_width


SCHEMA = 'shared_current_native_v1'
PUBLIC_FEATURES = 1832
SHARED_FEATURES = PUBLIC_FEATURES + PRIVILEGED_FEATURES
ARRAYS = {'cards':10, 'skills':16, 'special':28}
REDACTED = {'deck':20}


def _validate_alignment(public,native):
    current=public[:8*frame_width(16,True)].reshape(8,-1)[-1]
    for seat,player in enumerate(native.players):
        token=current[seat*10:seat*10+10]
        if token[0]==0: continue
        if token[0]!=1: raise ValueError('invalid public fighter presence')
        for column,name in ((3,'dir'),(4,'char'),(5,'act'),(6,'act_block'),(9,'frame')):
            if token[column]!=np.float32(player[name]):
                raise ValueError('public/native fighter state mismatch: '+name)
        world=current[frame_width(16)+2*seat:frame_width(16)+2*seat+2]
        if not np.array_equal(world,np.asarray((player['x'],player['y']),np.float32)):
            raise ValueError('public/native world position mismatch')


def _entity(entity, names):
    # Whitelist only existing current-state fields. Incidental input buffers,
    # RNG snapshots or future annotations cannot enter through arbitrary keys.
    result = {name:entity[name] for name in names}
    for kind in ('hitarea', 'attackarea'):
        result[kind] = tuple(tuple(box) for box in entity[kind])
    return result


def sanitize_native(observation):
    """Independent seat-relative copy of current state, no truncation/defaults."""
    if len(observation.players)!=2:
        raise ValueError('shared native input requires both current players')
    players=[]
    for seat,player in enumerate(observation.players):
        result=_entity(player,FIGHTER_NAMES)
        for key,length in ARRAYS.items():
            result[key]=tuple(player[key])
            if len(result[key])!=length: raise ValueError('invalid shared native array length')
        result.update({key:(0,)*length for key,length in REDACTED.items()})
        # Own already-consumed key counters are past/current self information,
        # not the opponent's private decision. Preserve them for prefix checks.
        result['keys']=tuple(player['keys']) if seat==0 else (0,)*10
        if len(result['keys'])!=10: raise ValueError('invalid own input-counter width')
        result['objects']=tuple(_entity(obj,OBJECT_NAMES) for obj in player['objects'])
        players.append(result)
    return PrivilegedObservation({key:observation.world[key] for key in WORLD_NAMES},tuple(players))


def assemble_shared(public, native, *, public_frame):
    """Preserve the1832prefix; append only SAME-frame sanitized native context."""
    if (not isinstance(public,np.ndarray) or public.shape!=(PUBLIC_FEATURES,) or
            public.dtype!=np.float32 or not np.isfinite(public).all()):
        raise ValueError('shared v1 requires the unchanged1832 public prefix')
    if type(public_frame) is not int or public_frame<0 or native.world['frame']!=public_frame:
        raise ValueError('public/native frame mismatch; do not pair stale captures')
    _validate_alignment(public,native)
    result=np.concatenate((public,encode_privileged(sanitize_native(native))))
    if result.shape!=(SHARED_FEATURES,): raise ValueError('shared observation shape mismatch')
    return result


def decode_shared(values, *, schema):
    if schema!=SCHEMA:
        raise ValueError('explicit shared observation version required')
    if (not isinstance(values,np.ndarray) or values.dtype!=np.float32 or
            values.shape!=(SHARED_FEATURES,) or not np.isfinite(values).all()):
        raise ValueError('invalid shared current-state tensor')
    current=decode_privileged(values[PUBLIC_FEATURES:])
    for seat,player in enumerate(current.players):
        if any(any(player[key]) for key in REDACTED) or (seat==1 and any(player['keys'])):
            raise ValueError('private input/deck channels were not redacted')
    _validate_alignment(values[:PUBLIC_FEATURES],current)
    return current


@dataclass(frozen=True)
class SharedNativeContract:
    schema: str = SCHEMA
    observation_frames: int = 1
    additional_reaction_frames: int = 0
    control: str = 'UNASSIGNED_NO_TRAINING_ATTACHMENT'

    def __post_init__(self):
        if (self.schema!=SCHEMA or self.observation_frames!=1 or
                self.additional_reaction_frames!=0 or
                self.control!='UNASSIGNED_NO_TRAINING_ATTACHMENT'):
            raise ValueError('shared v1 establishes information parity, not control takeover')

    def report(self):
        return dict(schema=self.schema,public_features=PUBLIC_FEATURES,
            native_features=PRIVILEGED_FEATURES,total_features=SHARED_FEATURES,
            bytes_per_observation=SHARED_FEATURES*4,
            additional_reaction_frames=0,
            redacted=['enemy_keys','deck'],
            own_inputs='existing native already-consumed counters plus original1832 own submitted history',
            pointers='retained by common lossless data contract; neural encoder masks pointer magnitudes',
            object_scope='complete existing Platform bounded lists; overflow is an error, not truncation',
            control=self.control,native_capture_verified=False,training_ready=False)

"""Versioned native contact sidecar; no pointer is a policy observation."""
import ctypes
import math
import struct
import bridge_shared

HEADER = struct.Struct('<8I')
EVENT_V1 = struct.Struct('<30I')
EVENT = struct.Struct('<34I')
EVENT_V4 = struct.Struct('<35I')
EVENT_V5 = struct.Struct('<52I')
CAPACITY = 2048


def signed(value):
    return ctypes.c_int32(value).value


def rate(value):
    if value == 0xFFFFFFFF:
        return None
    result=struct.unpack('<f',struct.pack('<I',value))[0]
    if not math.isfinite(result):
        raise ValueError('nonfinite native Rate')
    return result


def decode_event(values, version=3):
    if len(values) not in (30,34,35,52) or values[4] not in (1,2,3,4,5,6,7) or values[5] not in (0,1) or values[6] not in (0,1):
        raise ValueError('invalid native contact event')
    if values[4] in (4,5) and len(values)<34:
        raise ValueError('action/source identity requires combat ABI2+')
    identity = dict(source_instance=values[30] or None, attack_instance=values[31] or None,
                    lineage={0:'UNKNOWN',1:'DIRECT',2:'CHILD'}[values[32]]) if len(values)>=34 else dict(
                        source_instance=None,attack_instance=None,lineage='UNKNOWN')
    common = dict(frame=values[0]+(values[1]<<32),segment=values[2],order=values[3],
                  attacker_seat=values[5],source_address=values[7],**identity)
    if values[4] in (6,7):
        if version<(6 if values[4]==6 else 7) or len(values)!=52:
            raise ValueError('input samples require combat ABI6')
        return dict(**common,kind='NATIVE_INPUT_SCOPE' if values[4]==6 else 'NATIVE_INPUT_SAMPLE',native_input_scope=values[35],
                    buffered_inputs_before=tuple(signed(v) for v in values[36:44]),
                    native_inputs_before=tuple(signed(values[i]) for i in (19,20,13,14,15,16,17,18)))
    if values[4] == 4:
        inputs = tuple(signed(values[i]) for i in (19,20,13,14,15,16,17,18)) if version>=3 else None
        return dict(**common,kind='ACTION_TRANSITION',action_before=values[10],action_after=values[26],
                    accepted=bool(values[27]),caller=values[33],independent_input_segment=None,
                    native_inputs_before=inputs,
                    native_input_scope=values[35] or None if version>=5 and len(values)==52 else None,
                    buffered_inputs_before=tuple(signed(v) for v in values[36:44]) if version>=5 and len(values)==52 else None,
                    released_inputs_before=tuple(signed(v) for v in values[44:50]) if version>=5 and len(values)==52 else None,
                    command_combination_before=values[50] if version>=5 and len(values)==52 else None,
                    movement_combination_before=values[51] if version>=5 and len(values)==52 else None,
                    native_button_edges=tuple(i for i,v in enumerate(inputs[2:]) if v==1) if inputs else None)
    if values[4] == 5:
        return dict(**common,kind='OBJECT_CREATED',action=values[8],sequence=values[9],
                    parent_address=values[33])
    flags = values[34] if version>=4 and len(values)>=35 and values[34]!=0xFFFFFFFF else None
    first_ch = None
    if flags is not None and values[29]==1 and values[4]==1 and values[28]==1:
        first_ch = values[33]==0 and bool(flags & 0x20)
    return dict(**common,
        kind={1:'HIT_RESOLUTION',2:'BLOCK_HIGH_RESOLUTION',3:'BLOCK_LOW_RESOLUTION'}[values[4]],
        defender_seat=values[6],
        source_action=values[8], source_sequence=values[9], defender_action_before=values[10],
        defender_sequence_before=values[11], defender_flags_before=values[12],
        hp_before=signed(values[13]), hp_after=signed(values[14]),
        spirit_before=signed(values[15]), spirit_after=signed(values[16]),
        rate_before=rate(values[17]), rate_at_damage=rate(values[18]), rate_after=rate(values[19]),
        combo_count_before=values[20], combo_count_after=values[21],
        limit_before=values[22], limit_after=values[23], combo_flags_before=values[24],
        combo_flags_after=values[25], defender_action_after=values[26],
        native_return=values[27], source_collision_after=values[28], rate_read_count=values[29],
        independent_input_segment=None, punish_counter=first_ch,
        counter_hit_first=first_ch, combo_flags_at_damage=flags,
        punish_definition='FIRST_HIT_NATIVE_CH_PROXY',
        combo_count_at_damage=values[33] if version>=3 else None,
        hit_or_block={1:'HIT',2:'BLOCK',7:'ARMOR'}.get(values[28],'UNKNOWN'),
        evidence='NATIVE_RESOLVER_CALL', damage_rate=rate(values[18]))


class CombatCaptureClient:
    def __init__(self, pid):
        kernel = bridge_shared._kernel32
        self.handle = kernel.OpenFileMappingW(6, False, rf'Local\SokuRLCombat_{pid}')
        if not self.handle:
            raise OSError(ctypes.get_last_error(), 'native contact mapping unavailable')
        header_view = kernel.MapViewOfFile(self.handle,6,0,0,HEADER.size)
        if not header_view:
            kernel.CloseHandle(self.handle)
            raise OSError(ctypes.get_last_error(),'cannot map contact header')
        try:
            header=HEADER.unpack(ctypes.string_at(header_view,HEADER.size))
        finally:
            kernel.UnmapViewOfFile(header_view)
        valid = {1:EVENT_V1,2:EVENT,3:EVENT,4:EVENT_V4,5:EVENT_V5,6:EVENT_V5,7:EVENT_V5,8:EVENT_V5}
        if header[0]!=0x31564543 or header[1] not in valid or header[3]!=CAPACITY or header[2]!=valid[header[1]].size:
            kernel.CloseHandle(self.handle)
            raise RuntimeError('unsupported native combat ABI')
        self.event = valid[header[1]]
        self.version = header[1]
        self.native_input_counters = self.version>=8 and header[7]&3==3
        self.view = kernel.MapViewOfFile(self.handle, 6, 0, 0, HEADER.size+CAPACITY*self.event.size)
        if not self.view:
            kernel.CloseHandle(self.handle)
            raise OSError(ctypes.get_last_error(), 'cannot map native contacts')

    def drain(self):
        _,_,_,_,write,read,dropped,_ = HEADER.unpack(ctypes.string_at(self.view,HEADER.size))
        available = (write-read)&0xFFFFFFFF
        if dropped or available > CAPACITY:
            raise RuntimeError('native contact events lost; reward must not continue')
        events = [decode_event(self.event.unpack(ctypes.string_at(
            self.view+HEADER.size+((read+i)%CAPACITY)*self.event.size, self.event.size)),self.version)
                  for i in range(available)]
        ctypes.c_uint32.from_address(self.view+20).value = write
        return sorted(events,key=lambda event:(event['segment'],event['frame'],event['order']))

    def close(self):
        if getattr(self,'view',None):
            bridge_shared._kernel32.UnmapViewOfFile(self.view)
            self.view = None
        if getattr(self,'handle',None):
            bridge_shared._kernel32.CloseHandle(self.handle)
            self.handle = None
